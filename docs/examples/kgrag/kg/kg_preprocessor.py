import asyncio
import json
import os
import pandas as pd
import pickle
import sqlite3 as sql
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import TransientError
from tqdm import tqdm

from kg.kg_rep import *
from utils.logger import *
from utils.utils import *
from constants import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# Define a Semaphore to control concurrency (e.g., max 50 tasks at a time)
SEMAPHORE = asyncio.Semaphore(50)

class KG_Preprocessor():
    def __init__(self):
        self.driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    async def close(self):
        await self.driver.close()

    async def run_query_async(self, query, parameters=None, retries=5, delay=0.5):
        """Run a Cypher query in Neo4j."""
        # async with SEMAPHORE:
        #     async with self.driver.session() as session:
        #         result = await session.run(query, parameters)
        #         return await result.consume()
            
        for attempt in range(retries):
            try:
                async with SEMAPHORE:
                    async with self.driver.session() as session:
                        result = await session.run(query, parameters)
                        return [record async for record in result]
            except TransientError as e:
                if "DeadlockDetected" in str(e):
                    logger.warning(f"Deadlock detected, retrying {attempt + 1}/{retries}")
                    await asyncio.sleep(delay * (2 ** attempt))  # exponential backoff
                else:
                    raise e
        raise RuntimeError("Max retries reached.")

    def preprocess(self):
        raise NotImplementedError


class MovieKG_Preprocessor(KG_Preprocessor):
    def __init__(self):
        super().__init__()

        # JSON Database Paths
        KG_BASE_DIRECTORY = os.getenv("KG_BASE_DIRECTORY", "docs/examples/kgrag/dataset")
        JSON_PATHS = {
            "year_db": os.path.join(KG_BASE_DIRECTORY, "movie", "year_db.json"),
            "person_db": os.path.join(KG_BASE_DIRECTORY, "movie", "person_db.json"),
            "movie_db": os.path.join(KG_BASE_DIRECTORY, "movie", "movie_db.json"),
        }

        # ======== Load Data =========
        # Load Movie Data
        logger.info("Loading movie data...")
        with open(JSON_PATHS["movie_db"]) as f:
            self.movie_db = json.load(f)

        # Load Person Data
        logger.info("Loading person data...")
        with open(JSON_PATHS["person_db"]) as f:
            self.person_db = json.load(f)
            
        # Load Year Data (not always needed)
        logger.info("Loading year data...")
        with open(JSON_PATHS["year_db"]) as f:
            self.year_db = json.load(f)

    async def preprocess(self):
        await self.create_indices()
        # # ======== Insert Entities =========
        logger.info("Inserting entity: movies...")
        await self.insert_all_movies_async(self.movie_db)

        # logger.info("Inserting entity: persons...")
        await self.insert_all_persons_async(self.person_db)

        logger.info("Inserting entity: years...")
        await self.insert_all_years_async(range(1990, 2022))

        # # ======== Insert Relations =========
        logger.info("Inserting relations: movies...")
        await self.insert_all_movie_relations_async(self.movie_db)

        # logger.info("Inserting relations: persons...")
        await self.insert_all_person_relations_async(self.person_db)

        # logger.info("Inserting relations: years...")
        await self.insert_all_year_relations_async(self.movie_db, self.person_db)

        # Simulate the original KG is not complete - it's not necessary to keep all the information
        await self.sample_KG()

    async def create_indices(self):
        """Create indices for faster querying."""
        queries = [
            "CREATE INDEX IF NOT EXISTS FOR (m:Movie) ON (m.name)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.name)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Award) ON (a.name)",
            "CREATE INDEX IF NOT EXISTS FOR (g:Genre) ON (g.name)",
            "CREATE INDEX IF NOT EXISTS FOR (y:Year) ON (y.name)",
        ]
        for query in queries:
            await self.run_query_async(query)

    # =====================================================
    async def insert_entity_movie_async(self, movie):
        query = """
        MERGE (m:Movie {name: $name})
        SET m.release_date = $release_date, m.original_name = $original_name,
            m.original_language = $original_language, m.budget = $budget, m.revenue = $revenue, m.rating = $rating
        """
        await self.run_query_async(query, {
            "name": movie["title"].upper(), "original_name": movie.get("original_title"), "release_date": movie.get("release_date"),
            "original_language": movie.get("original_language"), 
            "budget": str(movie["budget"]) if "budget" in movie else None, 
            "revenue": str(movie["revenue"]) if "revenue" in movie else None,
            "rating": str(movie["rating"]) if "rating" in movie else None
        })

    async def insert_entity_person_async(self, person):
        query = """
        MERGE (p:Person {name: $name})
        SET p.birthday = $birthday
        """
        await self.run_query_async(query, {"name": person["name"].upper(), "birthday": person.get("birthday")})

    async def insert_entity_genre_async(self, genre):
        query = """
        MERGE (g:Genre {name: $name})
        """
        await self.run_query_async(query, {"name": genre["name"].upper()})

    async def insert_entity_year_async(self, year):
        query = """
        MERGE (y:Year {name: $year_name})
        """
        await self.run_query_async(query, {"year_name": str(year)})

    # =================================================
    async def insert_relation_person_award_async(self, person):
        query = """
        MATCH (p:Person {name: $person_name})
        MERGE (a:Award {type: $type, name: $name, year: $year_name})
        SET a.ceremony_number = $ceremony
        
        // Create correct relationship type
        MERGE (p)-[r:NOMINATED_FOR]->(a)
        SET r.winner = $winner, r.person = $person, r.movie = $movie
        
        // If they won, change the relationship type
        FOREACH (ignored IN CASE WHEN $winner = true THEN [1] ELSE [] END |
            MERGE (p)-[win:WON]->(a)
            SET win.winner = true, win.person = $person, win.movie = $movie
        )
        """
        for award in person.get("oscar_awards", []):  # This can handle future awards too
            movie = award["film"].upper() if award["film"] is not None else None
            await self.run_query_async(query, {
                "person_name": person["name"].upper(), 
                "type": "OSCAR AWARD", "name": award["category"].upper(), "year_name": str(award["year_ceremony"]),
                "ceremony": award["ceremony"],
                "winner": award["winner"], "person": award["name"].upper(), "movie": movie
            })
            
        
    async def insert_relation_cast_movie_async(self, movie):
        # Insert Cast Members
        query = """
        MATCH (m:Movie {name: $movie_name}), (p:Person {name: $person_name})
        MERGE (p)-[:ACTED_IN {character: $character, order: $order, gender: $gender}]->(m)
        """
        for cast in movie.get("cast", []):
            await self.run_query_async(query, {
                "movie_name": movie["title"].upper(), "person_name": cast["name"].upper(),
                "character": cast.get("character"), "order": cast.get("order"), "gender": cast.get("gender")
            })

    async def insert_relation_movie_director_async(self, movie):
        # Insert Directors
        query = """
        MATCH (m:Movie {name: $movie_name}), (p:Person {name: $person_name})
        MERGE (p)-[:DIRECTED]->(m)
        """
        for director in movie.get("crew", []):
            if director["job"] == "Director":
                await self.run_query_async(query, {"movie_name": movie["title"].upper(), "person_name": director["name"].upper()})

    async def insert_relation_movie_genres_async(self, movie):
        # Insert Genres
        query = """
        MATCH (m:Movie {name: $movie_name})
        MERGE (g:Genre {name: $genre_name})
        MERGE (m)-[:BELONGS_TO_GENRE]->(g)
        """
        for genre in movie.get("genres", []):
            await self.run_query_async(query, {"movie_name": movie["title"].upper(), "genre_name": genre["name"].upper()})

    async def insert_relation_movie_awards_async(self, movie):
        # Insert Awards
        query = """
        MATCH (m:Movie {name: $movie_name})
        MERGE (a:Award {type: $type, name: $name, year: $year})
        SET a.ceremony_number = $ceremony
        
        // Create correct relationship type
        MERGE (m)-[r:NOMINATED_FOR]->(a)
        SET r.winner = $winner, r.person = $person, r.movie = $movie
        
        // If they won, change the relationship type
        FOREACH (ignored IN CASE WHEN $winner = true THEN [1] ELSE [] END |
            MERGE (m)-[win:WON]->(a)
            SET win.winner = true, win.person = $person,win.movie = $movie
        )
        """
        for award in movie.get("oscar_awards", []):
            await self.run_query_async(query, {
                "movie_name": movie["title"].upper(),
                "type": "OSCAR AWARD", "name": award["category"].upper(), "year": str(award["year_ceremony"]),
                "ceremony": award["ceremony"],
                "winner": award["winner"], "person": award["name"].upper(), "movie": award["film"].upper()
            })

    async def insert_relation_year_movie_async(self, movie):
        # Extract year from release_date
        release_year = int(movie["release_date"][:4]) if movie.get("release_date") else None
        if release_year:
            query = """
            MATCH (m:Movie {name: $movie_name}), (y:Year {name: $year_name})
            MERGE (m)-[:RELEASED_IN]->(y)
            """
            await self.run_query_async(query, {"movie_name": movie["title"].upper(), "year_name": str(release_year)})

    async def insert_relation_year_award_async(self, item):
        # Create the HELD_IN relationship
        query = """
        MATCH (a:Award {type: $type, name: $name, year: $year_name}), (y:Year {name: $year_name})
        MERGE (a)-[:HELD_IN]->(y)
        """
        for award in item.get("oscar_awards", []):
            await self.run_query_async(query, {
                "type": "OSCAR AWARD", "name": award["category"].upper(), "year_name": str(award["year_ceremony"])
            })

    # =================================================
    async def insert_all_movies_async(self, movie_db):
        """Run async movie insertion with progress tracking."""
        tasks = [self.insert_entity_movie_async(movie) for movie in movie_db.values()]
        
        for task in tqdm(tasks, total=len(tasks), desc="Movies Entities Inserted"):
            await task

    async def insert_all_persons_async(self, person_db):
        """Run async movie insertion with progress tracking."""
        tasks = [self.insert_entity_person_async(person) for person in person_db.values()]
        
        for task in tqdm(tasks, total=len(tasks), desc="Persons Entities Inserted"):
            await task

    async def insert_all_years_async(self, years):
        """Run async movie insertion with progress tracking."""
        tasks = [self.insert_entity_year_async(year) for year in years]
        
        for task in tqdm(tasks, total=len(tasks), desc="Years Entities Inserted"):
            await task

    async def insert_movie_relations_async(self, movie):
        """Insert all relations for a single movie asynchronously."""
        await asyncio.gather(
            self.insert_relation_cast_movie_async(movie),
            self.insert_relation_movie_director_async(movie),
            self.insert_relation_movie_genres_async(movie),
            self.insert_relation_movie_awards_async(movie)
        )

    async def insert_all_movie_relations_async(self, movie_db):
        """Insert all movie relations with async progress tracking."""
        tasks = [self.insert_movie_relations_async(movie) for movie in movie_db.values()]

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Movies Relations Inserted"):
            await task

    async def insert_all_person_relations_async(self, person_db):
        """Insert all person relations with async progress tracking."""
        tasks = [self.insert_relation_person_award_async(person) for person in person_db.values()]

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Person Relations Inserted"):
            await task

    async def insert_all_year_relations_async(self, movie_db, person_db):
        """Insert all year relations with async progress tracking."""
        tasks = [self.insert_relation_year_movie_async(movie) for movie in movie_db.values()]
        tasks.extend([self.insert_relation_year_award_async(person) for person in person_db.values()])
        tasks.extend([self.insert_relation_year_award_async(movie) for movie in movie_db.values()])

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Year Relations Inserted"):
            await task

    async def sample_KG(self):
        labels = ["Movie", "Person", "Award", "Genre", "Year"]
        fractions = {"Movie": 0.6, "Person": 0.6, "Award": 1.0, "Genre": 1.0, "Year": 1.0}  # custom per label if needed
        
        for label in labels:
            query = f"""
            CALL {{
            WITH {fractions[label]} AS keep_fraction
            MATCH (n:{label})
            WHERE rand() > keep_fraction
            RETURN n
            }} 
            CALL {{
            WITH n
            DETACH DELETE n
            }} IN TRANSACTIONS OF 10000 ROWS
            """
            await self.run_query_async(query)


class SoccerKG_Preprocessor(KG_Preprocessor):
    def __init__(self):
        super().__init__()
        KG_BASE_DIRECTORY = os.getenv("KG_BASE_DIRECTORY", "dataset")
        file_name = 'soccer_team_match_stats.pkl'
        soccer_kg_file = os.path.join(KG_BASE_DIRECTORY, "sports", file_name)
        logger.info(f"Reading soccer KG from: {soccer_kg_file}")
        self.team_match_stats = pd.read_pickle(soccer_kg_file)
        self.team_match_stats = self.team_match_stats.where(pd.notnull(self.team_match_stats), None)
        logger.info("Soccer KG initialized ✅")

    async def preprocess(self):
        await self.create_indices()
        await self.insert_all_matches_async()

    async def create_indices(self):
        queries = [
            "CREATE INDEX IF NOT EXISTS FOR (t:Team) ON (t.name)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.name)",
            "CREATE INDEX IF NOT EXISTS FOR (l:League) ON (l.name)",
            "CREATE INDEX IF NOT EXISTS FOR (s:Season) ON (s.name)",
            "CREATE INDEX IF NOT EXISTS FOR (m:Match) ON (m.name)"
        ]
        for q in queries:
            await self.run_query_async(q)

    async def insert_all_matches_async(self):
        logger.info("Inserting Soccer Matches...")
        df = self.team_match_stats.reset_index()
        tasks = [self.insert_match_entity_async(row.to_dict()) for _, row in df.iterrows()]
        for task in tqdm(tasks, total=len(tasks), desc="Matches Inserted"):
            await task

    async def insert_entity_async(self, label, name):
        if pd.isna(name):
            return
        query = f"MERGE (n:{label} {{name: $name}})"
        await self.run_query_async(query, {"name": normalize_entity(name)})

    async def insert_match_entity_async(self, row):
        # Basic match info
        match_query = """
        MERGE (m:Match {name: $match_name})
        SET m.type = "soccer", m.date = $date, m.attendance = $att, m.notes = $notes;
        """
        match_name = f"{' '.join(row['game'].split(' ')[1:])} on {row['date'].date()}"
        await self.run_query_async(match_query, {
            "match_name": normalize_entity(match_name),
            "date": timestamp_to_text(row["date"], isDate=True),
            "att": normalize_value(row.get("Attendance")),
            "notes": normalize_value(row.get("Notes")),
        })

        # Related entities
        await self.insert_entity_async("Team", row["team"])
        await self.insert_entity_async("Team", row["opponent"])
        await self.insert_entity_async("Person", row["Captain"])
        await self.insert_entity_async("Person", row["Referee"])
        await self.insert_entity_async("League", row["league"])
        await self.insert_entity_async("Season", row["season"])

        # Relationships with venue as property
        rel_query = """
        MATCH (m:Match {name: $match_name})
        MATCH (t:Team {name: $team_name})
        MATCH (o:Team {name: $opp_name})
        MATCH (c:Person {name: $captain})
        MATCH (ref:Person {name: $referee})
        MATCH (l:League {name: $league})
        MATCH (s:Season {name: $season})
        MERGE (t)-[r:PLAYED_IN]->(m)
        SET r.result = $result, r.gf = $gf, r.ga = $ga, r.xg = $xg, r.xga = $xga, r.possession = $poss, r.formation = $formation, r.venue = $venue
        
        MERGE (c)-[:CAPTAINED]->(m)
        MERGE (ref)-[:REFEREED]->(m)
        MERGE (l)-[:HAS_MATCH]->(m)
        MERGE (s)-[:HAS_MATCH]->(m)
        """
        await self.run_query_async(rel_query, {
            "match_name": normalize_entity(match_name),
            "team_name": normalize_entity(row["team"]),
            "opp_name": normalize_entity(row["opponent"]),
            "captain": normalize_entity(row["Captain"]) if not pd.isna(row["Captain"]) else None,
            "referee": normalize_entity(row["Referee"]) if not pd.isna(row["Referee"]) else None,
            "league": normalize_entity(row["league"]),
            "season": normalize_entity(row["season"]),
            
            "result": normalize_value(row.get("result")),
            "gf": normalize_value(row.get("GF")),
            "ga": normalize_value(row.get("GA")),
            "xg": normalize_value(row.get("xG")),
            "xga": normalize_value(row.get("xGA")),
            "poss": normalize_value(row.get("Poss")),
            "formation": normalize_value(row.get("Formation")),
            "venue": normalize_value(row["venue"].lower())
        })


class NBAKG_Preprocessor(KG_Preprocessor):
    def __init__(self):
        super().__init__()
        KG_BASE_DIRECTORY = os.getenv("KG_BASE_DIRECTORY", "dataset")
        nba_kg_file = os.path.join(KG_BASE_DIRECTORY, "sports", "nba.sqlite")
        logger.info(f"Loading NBA SQLite DB from: {nba_kg_file}")
        conn = sql.connect(nba_kg_file)

        df_players = pd.read_sql("SELECT * FROM player", conn)
        df_info = pd.read_sql("SELECT * FROM common_player_info", conn)
        df_draft = pd.read_sql("SELECT * FROM draft_history", conn)
        df_players = df_players.merge(df_info, how="left", left_on="id", right_on="person_id", suffixes=('', '_y'))
        df_players = df_players.merge(df_draft, how="left", left_on="id", right_on="person_id", suffixes=('', '_y'))
        df_players["team_id"] = df_players["team_id"].astype("Int64").astype("str")
        df_players["from_year"] = df_players["from_year"].astype("Int64")
        df_players["to_year"] = df_players["to_year"].astype("Int64")
        df_players = df_players.where(pd.notnull(df_players), None)
        self.df_players = df_players

        df_drafts = pd.read_sql("SELECT * FROM draft_history", conn)
        df_drafts = df_drafts.where(pd.notnull(df_drafts), None)
        self.df_drafts = df_drafts
    
        df_matches = pd.read_sql("SELECT * FROM game", conn)
        df_summary = pd.read_sql("SELECT * FROM game_summary", conn)
        df_info = pd.read_sql("SELECT * FROM game_info", conn)
        df_score = pd.read_sql("SELECT * FROM line_score", conn)
        df_stats = pd.read_sql("SELECT * FROM other_stats", conn)
        df_matches = df_matches.merge(df_summary, on="game_id", how="left", suffixes=('', '_y'))
        df_matches = df_matches.merge(df_info, on="game_id", how="left", suffixes=('', '_y'))
        df_matches = df_matches.merge(df_score, on="game_id", how="left", suffixes=('', '_y'))
        df_matches = df_matches.merge(df_stats, on="game_id", how="left", suffixes=('', '_y'))
        df_matches = df_matches.where(pd.notnull(df_matches), None)
        self.df_matches = df_matches

        df_teams = pd.read_sql("SELECT * FROM team", conn)
        df_details = pd.read_sql("SELECT * FROM team_details", conn)
        df_history = pd.read_sql("SELECT * FROM team_history", conn)
        df_teams = df_teams.merge(df_details, how="left", left_on="id", right_on="team_id", suffixes=('', '_y'))
        df_teams = df_teams.merge(df_history, how="left", on="team_id", suffixes=('', '_y'))
        # The team table is not a complete list, need to infer from the matches
        home_teams = df_matches[["team_id_home", "team_abbreviation_home", "team_name_home"]].rename(
            columns={"team_id_home": "id", "team_abbreviation_home": "abbreviation", "team_name_home": "full_name"}
        )
        away_teams = df_matches[["team_id_away", "team_abbreviation_away", "team_name_away"]].rename(
            columns={"team_id_away": "id", "team_abbreviation_away": "abbreviation", "team_name_away": "full_name"}
        )
        teams_from_matches = pd.concat([home_teams, away_teams], ignore_index=True).drop_duplicates(subset=["id"])
        teams_from_matches = teams_from_matches.loc[:, ~teams_from_matches.columns.duplicated()]
        df_teams = df_teams.loc[:, ~df_teams.columns.duplicated()]
        df_teams_extended = pd.concat([df_teams, teams_from_matches], ignore_index=True).drop_duplicates(subset=["id"])
        df_teams_extended = df_teams_extended.where(pd.notnull(df_teams_extended), None)
        self.df_teams = df_teams_extended

        df_officials = pd.read_sql("SELECT * FROM officials", conn)
        df_officials = df_officials.where(pd.notnull(df_officials), None)
        self.df_officials = df_officials

    async def create_indices(self):
        queries = [
            "CREATE INDEX IF NOT EXISTS FOR (s:Season) ON (s.name)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.name)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Team) ON (t.name)",
            "CREATE INDEX IF NOT EXISTS FOR (m:Match) ON (m.name)",
        ]
        for q in queries:
            await self.run_query_async(q)

    async def preprocess(self):
        await self.create_indices()

        logger.info("Inserting Players...")
        await self.insert_players()

        logger.info("Inserting Officials...")
        await self.insert_officials()

        logger.info("Inserting Teams...")
        await self.insert_teams()

        logger.info("Inserting Matches...")
        await self.insert_matches()

        logger.info("Inserting Sesons...")
        await self.insert_seasons()
        
        logger.info("Inserting Person and Team Relations...")
        await self.insert_person_team_realtions()

        logger.info("Inserting Person and Match Relations...")
        await self.insert_person_match_realtions()

        logger.info("Inserting Match and Team Relations...")
        await self.insert_match_and_team_relations()

    async def insert_players(self):
        tasks = [self.insert_player_async(row.to_dict()) for _, row in self.df_players.iterrows()]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Players Inserted"):
            await task

    async def insert_teams(self):
        tasks = [self.insert_team_async(row.to_dict()) for _, row in self.df_teams.iterrows()]
        for task in tqdm(tasks, total=len(tasks), desc="Teams Inserted"):
            await task

    async def insert_matches(self):
        tasks = [self.insert_match_async(row.to_dict()) for _, row in self.df_matches.iterrows()]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Matches Inserted"):
            await task

    async def insert_seasons(self):
        tasks = [self.insert_season_async(row.to_dict()) for _, row in self.df_matches.iterrows()]
        for task in tqdm(tasks, total=len(tasks), desc="Seasons Inserted"):
            await task

    async def insert_officials(self):
        tasks = [self.insert_official_async(row.to_dict()) for _, row in self.df_officials.iterrows()]
        for task in tqdm(tasks, total=len(tasks), desc="Officials Inserted"):
            await task

    async def insert_player_async(self, row):
        query = """
        MERGE (p:Person {id: $id})
        SET p.name = $name,
            p.birthdate = $birthdate,
            p.school = $school,
            p.country = $country,
            p.height = $height,
            p.weight = $weight,
            p.active = $active
        """
        await self.run_query_async(query, {
            "id": row['id'],
            "name": normalize_entity(row["full_name"]),
            "birthdate": normalize_value(row["birthdate"]),
            "school": normalize_value(row.get("school")),
            "country": normalize_value(row.get("country")),
            "height": normalize_value(row.get("height")),
            "weight": normalize_value(row.get("weight")),
            "active": normalize_value(str(bool(row["is_active"])))
        })

    async def insert_official_async(self, row):
        query = """
        MERGE (p:Person {id: $id})
        SET p.name = $name
        """
        await self.run_query_async(query, {
            "id": row['official_id'],
            "name": normalize_entity(f"{row["first_name"]} {row["last_name"]}"),
        })

    async def insert_team_async(self, row):
        query = """
        MERGE (t:Team {id: $id})
        SET t.type = "basketball",
            t.name = $team_name,
            t.abbreviation = $abbreviation,
            t.city = $city,
            t.state = $state,
            t.year_founded = $year_founded,
            t.arena = $arena,
            t.arena_capacity = $arena_capacity,
            t.owner = $owner,
            t.general_manager = $gm,
            t.head_coach = $coach,
            t.dleague_affiliation = $dleague
        """
        await self.run_query_async(query, {
            "id": row["id"],
            "team_name": normalize_entity(row["full_name"]),
            "abbreviation": normalize_value(row["abbreviation"]),
            "city": normalize_value(row["city"]),
            "state": normalize_value(row["state"]),
            "year_founded": normalize_value(row["year_founded"]),
            "arena": normalize_value(row["arena"]),
            "arena_capacity": normalize_value(row["arenacapacity"]),
            "owner": normalize_value(row["owner"]),
            "gm": normalize_value(row["generalmanager"]),
            "coach": normalize_value(row["headcoach"]),
            "dleague": normalize_value(row["dleagueaffiliation"]),
        })

    async def insert_match_async(self, row):
        match_name = f"{row['team_name_home']}-{row['team_name_away']} on {row['game_date']}"
        query = """
        MERGE (m:Match {id: $game_id})
        SET m.type = "basketball", m.name = $match_name, m.date = $game_date, m.home_team = $home_team_name, m.away_team = $away_team_name,
            m.points_home = $pts_home, m.points_away = $pts_away, m.winner = $winner, m.loser = $loser,
            m.season_type = $season_type, m._timestamp = $game_date
        """
        await self.run_query_async(query, {
            "match_name": normalize_entity(match_name),
            "game_id": row["game_id"],
            "game_date": normalize_value(row["game_date"]),
            "home_team_name": normalize_value(row["team_name_home"]),
            "away_team_name": normalize_value(row["team_name_away"]),
            "result": normalize_value(row["wl_home"]),
            "pts_home": normalize_value(row["pts_home"]),
            "pts_away": normalize_value(row["pts_away"]),
            "winner": normalize_value(row["team_name_home"] if row["wl_home"] == 'W' else row["team_name_away"]),
            "loser": normalize_value(row["team_name_home"] if row["wl_home"] == 'L' else row["team_name_away"]),
            "season_type": normalize_value(row["season_type"])
        })

    async def insert_season_async(self, row):
        query = """
        MERGE (s:Season {id: $id})
        SET s.name = $name, s.type = $season_type
        """
        await self.run_query_async(query, {
            "id": row["season_id"],
            "name": normalize_entity(row["season"]),
            "season_type": normalize_value(row["season_type"])
        })
    
    # === Relation insertion examples ===
    async def insert_person_team_realtions(self):
        tasks = [self.insert_person_team_realtions_async(row.to_dict()) for _, row in self.df_players.iterrows()]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Person-Team Inserted"):
            await task

    async def insert_person_match_realtions(self):
        tasks = [self.insert_person_match_realtions_async(row.to_dict()) for _, row in self.df_officials.iterrows()]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Person-Match Inserted"):
            await task

    async def insert_match_and_team_relations(self):
        tasks = [self.insert_match_and_team_relations_async(row.to_dict()) for _, row in self.df_matches.iterrows()]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Match-Team Inserted"):
            await task
            
    async def insert_person_team_realtions_async(self, row):
        query = """
        MATCH (p:Person {id: $p_id})
        MATCH (t:Team {id: $t_id})
        MERGE (p)-[r:JOINED {valid_from: $from_year, valid_until: $to_year}]->(t)
        SET r.position = $position,
            r.jersey = $jersey,
            r.season_exp = $season_exp

        FOREACH (_ IN CASE WHEN $draft_year IS NOT NULL AND $draft_round IS NOT NULL THEN [1] ELSE [] END |
          MERGE (p)-[d:DRAFTED_BY {year: $draft_year, round: $draft_round}]->(t)
          SET p.draft_number = $draft_number,
              p.organization = $organization
        )
        """
        await self.run_query_async(query, {
            "p_id": row["id"],
            "t_id": row["team_id"],
            "from_year": normalize_value(row["from_year"]),
            "to_year": normalize_value(row["to_year"]),
            "position": normalize_value(row["position"]),
            "jersey": normalize_value(row["jersey"]),
            "season_exp": normalize_value(row["season_exp"]),
            "draft_year": normalize_value(row["draft_year"]),
            "draft_round": normalize_value(row["draft_round"]),
            "draft_number": normalize_value(row["draft_number"]),
            "organization": normalize_value(row["organization"]),
        })

    async def insert_person_match_realtions_async(self, row):
        query = """
        MATCH (p:Person {id: $p_id})
        MATCH (m:Match {id: $m_id})
        MERGE (p)-[r:REFEREED]->(m)
        SET r.jersey = $jersey
        """
        await self.run_query_async(query, {
            "p_id": row["official_id"],
            "m_id": row["game_id"],
            "jersey": normalize_value(row["jersey_num"])
        })
        
    async def insert_match_and_team_relations_async(self, row):
        query = """
        MATCH (s:Season {id: $s_id})
        MATCH (m:Match {id: $m_id})
        MATCH (home:Team {id: $team_id_home})
        MATCH (away:Team {id: $team_id_away})
        MERGE (s)-[:HAS_MATCH]->(m)
        
        MERGE (home)-[r1:PLAYED_IN]->(m)
        SET r1.result = $wl_home, r1.points = $pts_home, r1.venue = "home"
    
        MERGE (away)-[r2:PLAYED_IN]->(m)
        SET r2.result = $wl_away, r2.points = $pts_away, r2.venue = "away"
        """
        await self.run_query_async(query, {
            "s_id": row["season_id"],
            "m_id": row["game_id"],
            "team_id_home": row["team_id_home"],
            "team_id_away": row["team_id_away"],
            "wl_home": normalize_value(row["wl_home"]),
            "pts_home": normalize_value(row["pts_home"]),
            "wl_away": normalize_value(row["wl_away"]),
            "pts_away": normalize_value(row["pts_away"])
        })


class MusicKG_Preprocessor(KG_Preprocessor):
    def __init__(self):
        super().__init__()

        self.batch_size = 10000

        KG_BASE_DIRECTORY = os.getenv("KG_BASE_DIRECTORY", "../dataset")
        # Reading the artist dictionary
        artist_dict_path = os.path.join(KG_BASE_DIRECTORY, "music", "artist_dict_simplified.pickle")
        with open(artist_dict_path, 'rb') as file:
            self.artist_dict = pickle.load(file)
        
        # Reading the song dictionary
        song_dict_path = os.path.join(KG_BASE_DIRECTORY, "music", "song_dict_simplified.pickle")
        with open(song_dict_path, 'rb') as file:
            self.song_dict = pickle.load(file)
        
        # Reading the Grammy DataFrame
        grammy_df_path = os.path.join(KG_BASE_DIRECTORY, "music", "grammy_df.pickle")
        with open(grammy_df_path, 'rb') as file:
            self.grammy_df = pickle.load(file)
        self.grammy_df = self.grammy_df.where(pd.notnull(self.grammy_df), None)
        
        # Reading the rank dictionary for Hot 100
        rank_dict_hot_path = os.path.join(KG_BASE_DIRECTORY, "music", "rank_dict_hot100.pickle")
        with open(rank_dict_hot_path, 'rb') as file:
            self.rank_dict_hot = pickle.load(file)
        
        # Reading the song dictionary for Hot 100
        song_dict_hot_path = os.path.join(KG_BASE_DIRECTORY, "music", "song_dict_hot100.pickle")
        with open(song_dict_hot_path, 'rb') as file:
            self.song_dict_hot = pickle.load(file)
        
        # Reading the artist work dictionary
        artist_work_dict_path = os.path.join(KG_BASE_DIRECTORY, "music", "artist_work_dict.pickle")
        with open(artist_work_dict_path, 'rb') as file:
            self.artist_work_dict = pickle.load(file)

    async def create_indices(self):
        queries = [
            "CREATE INDEX IF NOT EXISTS FOR (a:Artist) ON (a.name)",
            "CREATE INDEX IF NOT EXISTS FOR (s:Song) ON (s.name)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Award) ON (a.name)",
        ]
        for q in queries:
            await self.run_query_async(q)

    async def insert_artists(self):
        all_data = []
        member_edges = []
    
        for name, props in self.artist_dict.items():
            all_data.append({
                "name": normalize_entity(name),
                "birth_date": normalize_value(props.get("birth_date")),
                "end_date": normalize_value(props.get("end_date")),
                "country": normalize_value(props.get("country")),
                "type": normalize_value(props.get("type"))
            })
    
            for member in props.get("members", []):
                member_edges.append({
                    "band": normalize_entity(name),
                    "member": normalize_entity(member)
                })
    
        # Insert nodes in batches
        query = """
        UNWIND $batch AS row
        MERGE (a:Artist {name: row.name})
        SET a.birth_date = row.birth_date,
            a.end_date = row.end_date,
            a.country = row.country,
            a.type = row.type
        """
        for i in tqdm(range(0, len(all_data), self.batch_size), desc="Batch Insert Artists"):
            batch = all_data[i:i + self.batch_size]
            await self.run_query_async(query, {"batch": batch})
    
        # Insert MEMBER_OF relations in batches
        rel_query = """
        UNWIND $edges AS row
        MATCH (band:Artist {name: row.band})
        MATCH (member:Artist {name: row.member})
        MERGE (member)-[:MEMBER_OF]->(band)
        """
        for i in tqdm(range(0, len(member_edges), self.batch_size), desc="Insert Artist-Artist edges"):
            edge_batch = member_edges[i:i + self.batch_size]
            await self.run_query_async(rel_query, {"edges": edge_batch})

    async def insert_songs(self):
        all_data = []
        wrote_edges = []
    
        for name, props in self.song_dict.items():
            all_data.append({
                "name": normalize_entity(name),
                "author": normalize_entity(props["author"]),
                "date": normalize_value(props.get("date")),
                "country": normalize_value(props.get("country")),
            })
            if props.get("author"):
                wrote_edges.append({
                    "song": normalize_entity(name),
                    "author": normalize_entity(props["author"])
                })

        for i in tqdm(range(0, len(all_data), self.batch_size), desc="Batch Insert Songs"):
            batch = all_data[i:i + self.batch_size]
            await self.run_query_async("""
            UNWIND $batch AS row
            MERGE (s:Song {name: row.name, author: row.author})
            SET s.country = row.country
            """, {"batch": batch})
    
        for i in tqdm(range(0, len(wrote_edges), self.batch_size), desc="Insert Artist-Song edges"):
            edge_batch = wrote_edges[i:i + self.batch_size]
            await self.run_query_async("""
            UNWIND $edges AS row
            MATCH (s:Song {name: row.song, author: row.author}), (a:Artist {name: row.author})
            MERGE (a)-[:WROTE]->(s)
            """, {"edges": edge_batch})

    async def insert_grammy_awards(self):
        awards = []
        artist_wins = []
        song_wins = []
    
        for _, row in self.grammy_df.iterrows():
            title = normalize_entity(row["title"])
            category = normalize_value(row["category"])
            year = normalize_value(row["year"])
            nominee = normalize_entity(row["nominee"])
            artist = normalize_entity(row["artist"])
    
            awards.append({"title": title, "category": category, "year": year})
    
            if row["artist"]:
                song_wins.append({"title": title, "category": category, "nominee": nominee, "author": artist})
            else:
                if row["nominee"] in self.song_dict:
                    song_wins.append({"title": title, "category": category, "nominee": nominee, "author": None})
                if row["nominee"] in self.artist_dict:
                    artist_wins.append({"title": title, "category": category, "nominee": nominee})

        for i in tqdm(range(0, len(awards), self.batch_size), desc="Batch Insert Awards"):
            batch = awards[i:i + self.batch_size]
            await self.run_query_async("""
            UNWIND $batch AS row
            MERGE (a:Award {name: row.title, category: row.category})
            SET a.year = row.year
            """, {"batch": batch})
    
        if artist_wins:
            for i in tqdm(range(0, len(artist_wins), self.batch_size), desc="Insert Artist-Award edges"):
                batch = artist_wins[i:i + self.batch_size]
                await self.run_query_async("""
                UNWIND $batch AS row
                MATCH (a:Award {name: row.title, category: row.category})
                MERGE (n:Artist {name: row.nominee})
                MERGE (n)-[:WON]->(a)
                """, {"batch": batch})
    
        if song_wins:
            for i in tqdm(range(0, len(song_wins), self.batch_size), desc="Insert Song-Award edges"):
                batch = song_wins[i:i + self.batch_size]
                await self.run_query_async("""
                UNWIND $batch AS row
                MATCH (a:Award {name: row.title, category: row.category})
                FOREACH (_ IN CASE WHEN row.author IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (n:Song {name: row.nominee, author: row.author})
                    MERGE (n)-[:WON]->(a)
                )
                FOREACH (_ IN CASE WHEN row.author IS NULL THEN [1] ELSE [] END |
                    MERGE (n:Song {name: row.nominee})
                    MERGE (n)-[:WON]->(a)
                )
                MERGE (n)-[:WON]->(a)
                """, {"batch": batch})

    async def insert_artist_works(self):
        work_rels = []
        for artist, works in self.artist_work_dict.items():
            for work in works:
                work_rels.append({
                    "artist": normalize_entity(artist),
                    "song": normalize_entity(work)
                })
    
        if work_rels:
            for i in tqdm(range(0, len(work_rels), self.batch_size), desc="Insert Artist-Song edges"):
                batch = work_rels[i:i + self.batch_size]
                await self.run_query_async("""
                UNWIND $batch AS row
                MATCH (a:Artist {name: row.artist}), (s:Song {name: row.song, author: row.artist})
                MERGE (a)-[:CREATED]->(s)
                """, {"batch": batch})

    async def sample_KG(self):
        labels = ["Song", "Artist", "Award"]
        fractions = {"Song": 0.2, "Artist": 0.2, "Award": 1.0}  # custom per label if needed
        
        for label in labels:
            query = f"""
            CALL {{
              WITH {fractions[label]} AS keep_fraction
              MATCH (n:{label})
              WHERE rand() > keep_fraction
              RETURN n
            }} 
            CALL {{
              WITH n
              DETACH DELETE n
            }} IN TRANSACTIONS OF 10000 ROWS
            """
            await self.run_query_async(query)

    async def preprocess(self):
        await self.create_indices()
        await self.insert_artists()
        await self.insert_songs()
        await self.insert_grammy_awards()
        await self.insert_artist_works()

        # The original KG is too large, and it's not necessary to keep all the information
        await self.sample_KG()


class MultiTQKG_Preprocessor(KG_Preprocessor):
    def __init__(self):
        super().__init__()

        KG_BASE_DIRECTORY = os.getenv("KG_BASE_DIRECTORY", "./dataset")
        self.triplet_path = os.path.join(KG_BASE_DIRECTORY, "MultiTQ/kg/full.txt")
        self.entity2id_path = os.path.join(KG_BASE_DIRECTORY, "MultiTQ/kg/entity2id.json")
        self.relation2id_path = os.path.join(KG_BASE_DIRECTORY, "MultiTQ/kg/relation2id.json")
        self.time2id_path = os.path.join(KG_BASE_DIRECTORY, "MultiTQ/kg/ts2id.json")

        # Load triplets
        self.triple_list = []
        with open(self.triplet_path, 'r') as file:
            for line in file:
                triplets = line.strip().replace("_", " ").split('\t')
                self.triple_list.append(triplets)
        self.triple_list = [[item.replace('_', ' ') for item in sublist] for sublist in self.triple_list]

        print("Custom KG initialized ✅")

    async def preprocess(self):
        await self.create_indices()
        await self.insert_all_facts_async()

    async def create_indices(self):
        queries = [
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (r:Relation) ON (r.name)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Time) ON (t.name)"
        ]
        for q in queries:
            await self.run_query_async(q)

    async def insert_entity_async(self, label, name):
        if pd.isna(name):
            return
        query = f"MERGE (n:{label} {{name: $name}})"
        await self.run_query_async(query, {"name": normalize_entity(name)})

    async def insert_fact_async(self, triple):
        h, r, t = triple[:3]
        time = triple[3] if len(triple) > 3 else None

        await self.insert_entity_async("Entity", h)
        await self.insert_entity_async("Entity", t)

        query = f"""
        MATCH (h:Entity {{name: $head}})
        MATCH (t:Entity {{name: $tail}})
        MERGE (h)-[rel:{normalize_relation(r)}]->(t)
        SET rel.time = $time
        """
        await self.run_query_async(query, {
            "head": normalize_entity(h),
            "tail": normalize_entity(t),
            "time": time
        })

    async def insert_all_facts_async(self):
        print("Inserting facts into the KG...")
        tasks = [self.insert_fact_async(triple) for triple in self.triple_list]
        for task in tqdm(tasks, total=len(tasks), desc="Facts Inserted"):
            await task


class TimeQuestionsKG_Preprocessor(KG_Preprocessor):
    def __init__(self):
        super().__init__()

        self.batch_size = 10000

        KG_BASE_DIRECTORY = os.getenv("KG_BASE_DIRECTORY", "./dataset")
        self.triplet_path = os.path.join(KG_BASE_DIRECTORY, "TimeQuestions/kg/full.txt")
        self.entity2id_path = os.path.join(KG_BASE_DIRECTORY, "TimeQuestions/kg/entity2id.json")
        self.relation2id_path = os.path.join(KG_BASE_DIRECTORY, "TimeQuestions/kg/relation2id.json")
        self.time2id_path = os.path.join(KG_BASE_DIRECTORY, "TimeQuestions/kg/ts2id.json")

        # Load triplets
        self.triple_list = []
        with open(self.triplet_path, 'r') as file:
            for line in file:
                triplets = line.strip().replace("_", " ").split('\t')
                self.triple_list.append(triplets)
        self.triple_list = [[item.replace('_', ' ') for item in sublist] for sublist in self.triple_list]

        print("Custom KG initialized ✅")

    async def preprocess(self):
        await self.create_indices()
        await self.insert_all_facts_async()

    async def create_indices(self):
        queries = [
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (r:Relation) ON (r.name)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Time) ON (t.name)"
        ]
        for q in queries:
            await self.run_query_async(q)

    async def insert_entity_async(self, label, name):
        if pd.isna(name):
            return
        query = f"MERGE (n:{label} {{name: $name}})"
        await self.run_query_async(query, {"name": normalize_entity(name)})

    # async def insert_fact_async(self, triple):
    #     h, r, t = triple[:3]
    #     from_time, to_time = (triple[3], triple[4]) if len(triple) > 4 else (None, None)

    #     await self.insert_entity_async("Entity", h)
    #     await self.insert_entity_async("Entity", t)

    #     query = f"""
    #     MATCH (h:Entity {{name: $head}})
    #     MATCH (t:Entity {{name: $tail}})
    #     MERGE (h)-[rel:{normalize_relation(r)}]->(t)
    #     SET rel.valid_from_year = $from_time, rel.valid_until_year = $to_time
    #     """
    #     await self.run_query_async(query, {
    #         "head": normalize_entity(h),
    #         "tail": normalize_entity(t),
    #         "from_time": from_time,
    #         "to_time": to_time
    #     })
    async def insert_facts_batch_async(self, triples):
        # Preprocess and normalize
        batch = []
        for triple in triples:
            h, r, t = triple[:3]
            from_time, to_time = (triple[3], triple[4]) if len(triple) > 4 else (None, None)
            batch.append({
                "head": normalize_entity(h),
                "tail": normalize_entity(t),
                "rel_type": normalize_relation(r),
                "from_time": from_time,
                "to_time": to_time
            })

        # Cypher query using UNWIND
        query = """
        UNWIND $batch AS row
        MERGE (h:Entity {name: row.head})
        MERGE (t:Entity {name: row.tail})
        MERGE (h)-[rel:REL]->(t)
        SET rel.valid_from_year = row.from_time,
            rel.valid_until_year = row.to_time
        """

        # Since we can't parametrize relationship types dynamically in Cypher,
        # we group by relation type and run separate queries
        from collections import defaultdict
        grouped = defaultdict(list)
        for row in batch:
            grouped[row["rel_type"]].append(row)

        for rel_type, rows in tqdm(grouped.items(), desc="Inserting Grouped Edges"):
            rel_query = query.replace("REL", rel_type)
            await self.run_query_async(rel_query, {"batch": rows})

    async def insert_all_facts_async(self):
        print("Inserting facts into the KG...")
        await self.insert_facts_batch_async(self.triple_list)