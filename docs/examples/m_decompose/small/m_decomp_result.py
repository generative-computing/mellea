# pytest: ollama, llm

import textwrap

import mellea

if __name__ == "__main__":
    m = mellea.start_session()

    # 1. Create a detailed shopping list with estimated costs for decorations, food, drinks, party favors, and a cake that serves at least 20 people, ensuring the total budget stays within $400. - - SHOPPING_LIST
    shopping_list = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to create a detailed shopping list with estimated costs for decorations, food, drinks, party favors, and a cake that serves at least 20 people, ensuring the total budget stays within $400. Follow these steps to accomplish your task:

         1. **Understand the Requirements**:
            - The party is for an 8-year-old daughter who loves unicorns.
            - The party theme colors are pink, purple, and gold.
            - The party will have 15 kids from her class, but the cake should serve at least 20 people.
            - The total budget should not exceed $400.
            - Follow basic food safety guidelines, especially for kids with common allergies like nuts and dairy.

         2. **Categories to Include**:
            - **Decorations**: Unicorn-themed decorations in pink, purple, and gold.
            - **Food**: Finger foods and snacks suitable for kids.
            - **Drinks**: Beverages that are kid-friendly.
            - **Party Favors**: Small gifts or treats for each child.
            - **Cake**: A unicorn-themed cake that serves at least 20 people.

         3. **Estimate Costs**:
            - Research and list the estimated costs for each item.
            - Ensure the total cost of all items does not exceed $400.

         4. **Create the Shopping List**:
            - Organize the list by categories (decorations, food, drinks, party favors, cake).
            - Include the name of each item, the estimated cost, and the quantity needed.
            - Ensure all items fit within the unicorn theme and the specified colors.

         5. **Review the List**:
            - Double-check that all necessary items are included.
            - Verify that the total estimated cost is within the $400 budget.

         6. **Output the Shopping List**:
            - Present the shopping list in a clear and organized format.
            - Include the estimated cost for each item and the total estimated cost.

         Here is an example structure to guide your writing:
         - **Decorations**:
            - Item 1: [Description], Quantity: [Number], Estimated Cost: [$Amount]
            - Item 2: [Description], Quantity: [Number], Estimated Cost: [$Amount]
         - **Food**:
            - Item 1: [Description], Quantity: [Number], Estimated Cost: [$Amount]
            - Item 2: [Description], Quantity: [Number], Estimated Cost: [$Amount]
         - **Drinks**:
            - Item 1: [Description], Quantity: [Number], Estimated Cost: [$Amount]
            - Item 2: [Description], Quantity: [Number], Estimated Cost: [$Amount]
         - **Party Favors**:
            - Item 1: [Description], Quantity: [Number], Estimated Cost: [$Amount]
            - Item 2: [Description], Quantity: [Number], Estimated Cost: [$Amount]
         - **Cake**:
            - Item 1: [Description], Quantity: [Number], Estimated Cost: [$Amount]

         Ensure that the shopping list is comprehensive and meets all the specified requirements.
         """.strip()
        ),
        requirements=[
            "The shopping list must include decorations, food, drinks, party favors, and a cake that serves at least 20 people",
            "Everything must stay within a total budget of about ~$400",
            "Follow basic food safety guidelines especially for kids with common allergies like nuts and dairy",
            "The party theme colors should be pink, purple, and gold to match the unicorn decorations",
        ],
    )
    assert shopping_list.value is not None, (
        'ERROR: task "shopping_list" execution failed'
    )

    # 2. Suggest 5-7 age-appropriate party games or activities that fit the unicorn theme and can keep the kids entertained for about 3 hours. - - PARTY_GAMES
    party_games = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to suggest 5-7 age-appropriate party games or activities that fit the unicorn theme and can keep the kids entertained for about 3 hours. Follow these steps to accomplish your task:

         1. **Understand the Theme and Audience**:
            - The party theme is unicorns, and the target audience is 8-year-old kids.
            - The party should be fun, engaging, and age-appropriate.
            - The theme colors are pink, purple, and gold.

         2. **Consider the Duration**:
            - The activities should keep the kids entertained for about 3 hours.
            - Plan a mix of active and passive games to maintain engagement.

         3. **Brainstorm Activity Ideas**:
            - Think of games and activities that fit the unicorn theme.
            - Ensure the activities are safe, fun, and suitable for 8-year-olds.
            - Consider both indoor and outdoor activities in case of rain.

         4. **List the Activities**:
            - Provide a list of 5-7 activities with brief descriptions.
            - Include any necessary materials or preparations for each activity.
            - Ensure the activities are varied to keep the kids interested.

         5. **Review the Shopping List**:
            - Refer to the shopping list created in the previous step to ensure you have all the necessary materials for the activities:
            <shopping_list>
            {{SHOPPING_LIST}}
            </shopping_list>

         6. **Finalize the Activities**:
            - Ensure the activities are feasible within the budget and theme.
            - Make sure the activities are safe and consider any common allergies or dietary restrictions.

         Here is an example structure to guide your suggestions:
         - **Activity 1**: [Description]
         - **Activity 2**: [Description]
         - **Activity 3**: [Description]
         - **Activity 4**: [Description]
         - **Activity 5**: [Description]
         - **Activity 6**: [Description]
         - **Activity 7**: [Description]

         Ensure each activity is clearly described and fits the unicorn theme. You should write only the activities, do not include the guidance structure.
         """.strip()
        ),
        requirements=[
            "Suggest 5-7 age-appropriate party games or activities that fit the unicorn theme and can keep the kids entertained for about 3 hours",
            "The party theme colors should be pink, purple, and gold to match the unicorn decorations",
        ],
        user_variables={"SHOPPING_LIST": shopping_list.value},
    )
    assert party_games.value is not None, 'ERROR: task "party_games" execution failed'

    # 3. Draft a timeline for the party day starting from 2 hours before guests arrive until cleanup is done, including food prep, decoration setup, and scheduling of activities. - - PARTY_TIMELINE
    party_timeline = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to draft a detailed timeline for the birthday party day, starting from 2 hours before guests arrive until cleanup is done. The timeline should include food prep, decoration setup, and scheduling of activities, ensuring the party runs smoothly and stays within the allocated time.

         To accomplish this, follow these steps:

         1. **Review the Shopping List and Party Games**:
            First, review the shopping list and party games that have been prepared in the previous steps. This will help you understand the tasks that need to be scheduled and the activities that will take place during the party.
            <shopping_list>
            {{SHOPPING_LIST}}
            </shopping_list>
            <party_games>
            {{PARTY_GAMES}}
            </party_games>

         2. **Plan the Timeline**:
            Create a timeline that starts 2 hours before the guests arrive and ends with the cleanup after the party. Include the following key elements:
            - **Decoration Setup**: Schedule the setup of decorations, ensuring they are in place before guests arrive.
            - **Food Prep**: Plan the preparation of food and drinks, including any last-minute tasks that need to be done just before serving.
            - **Activity Schedule**: Allocate time slots for each of the 5-7 party games or activities, ensuring they are spaced out appropriately to keep the kids entertained for about 3 hours.
            - **Food and Cake Serving**: Schedule the serving of food and cake, making sure it fits well within the activity timeline.
            - **Cleanup**: Include a time slot for cleanup after the party, ensuring all tasks are completed efficiently.

         3. **Ensure Logical Flow**:
            Make sure the timeline flows logically, with each task leading smoothly into the next. Consider the time required for each activity and the transitions between them.

         4. **Include Buffer Time**:
            Add buffer time between activities to account for any unexpected delays or transitions.

         5. **Finalize the Timeline**:
            Compile all the scheduled tasks into a clear and concise timeline. Ensure it is easy to follow and includes all necessary details.

         Here is an example structure to guide your writing:
         - **2 Hours Before Guests Arrive**: Start decoration setup and initial food prep.
         - **1.5 Hours Before Guests Arrive**: Complete decoration setup and continue food prep.
         - **1 Hour Before Guests Arrive**: Final touches on decorations and food prep.
         - **Guests Arrive**: Welcome guests and begin the first activity.
         - **Activity 1**: [Describe the activity and its duration]
         - **Activity 2**: [Describe the activity and its duration]
         - **Serve Food**: [Schedule the serving of food]
         - **Activity 3**: [Describe the activity and its duration]
         - **Serve Cake**: [Schedule the serving of cake]
         - **Activity 4**: [Describe the activity and its duration]
         - **Activity 5**: [Describe the activity and its duration]
         - **Cleanup**: [Schedule the cleanup tasks]

         Ensure that each time slot is clearly defined and that the timeline is easy to follow. You should write only the timeline, do not include the guidance structure.
         """.strip()
        ),
        requirements=[
            "Write out a timeline for the party day starting from 2 hours before guests arrive until cleanup is done showing exactly when to do food prep, decoration setup, when each activity should happen, and when to serve food and cake",
            "Follow basic food safety guidelines especially for kids with common allergies like nuts and dairy",
        ],
        user_variables={
            "SHOPPING_LIST": shopping_list.value,
            "PARTY_GAMES": party_games.value,
        },
    )
    assert party_timeline.value is not None, (
        'ERROR: task "party_timeline" execution failed'
    )

    # 4. Create a cute invitation text that includes all the important details like date, time, location, RSVP information, and any special instructions about allergies or what kids should bring or wear. - - INVITATION_TEXT
    invitation_text = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to create a cute invitation text for an 8-year-old's unicorn-themed birthday party. The invitation should include all the important details and follow the party's theme colors: pink, purple, and gold.

         To accomplish this, follow these steps:

         1. **Gather Party Details**:
            - Review the shopping list, party games, and timeline created in the previous steps to gather all necessary information:
            <shopping_list>
            {{SHOPPING_LIST}}
            </shopping_list>
            <party_games>
            {{PARTY_GAMES}}
            </party_games>
            <party_timeline>
            {{PARTY_TIMELINE}}
            </party_timeline>

         2. **Include Essential Information**:
            - **Date and Time**: Clearly state the date and time of the party.
            - **Location**: Provide the address and any specific instructions for getting to the party location.
            - **RSVP Information**: Include a contact email or phone number for RSVP and any deadline for responding.
            - **Special Instructions**: Mention any allergies, what kids should bring or wear, and any other important notes.

         3. **Theme and Tone**:
            - Use a cute and playful tone that fits the unicorn theme.
            - Incorporate the theme colors (pink, purple, and gold) into the design and wording of the invitation.

         4. **Format the Invitation**:
            - Use a clear and organized format, such as bullet points or paragraphs, to present the information.
            - Make sure the invitation is easy to read and visually appealing.

         5. **Example Structure**:
            - **Header**: A fun and colorful header that says "Unicorn Birthday Party!"
            - **Body**:
               - "You're invited to [Child's Name]'s Unicorn Birthday Party!"
               - "Date: [Date]"
               - "Time: [Time]"
               - "Location: [Location]"
               - "RSVP by [Date] to [Email/Phone Number]"
               - "Special Instructions: [Allergies, what to bring or wear, etc.]"
            - **Footer**: A cute closing line like "Hope to see you there for a magical time!"

         6. **Review and Finalize**:
            - Ensure all important details are included and the invitation is error-free.
            - Make any necessary adjustments to ensure the invitation is clear, cute, and fits the unicorn theme.

         Finally, write the cute invitation text based on the above guidelines and provide it as your answer.
         """.strip()
        ),
        requirements=[
            "Draft a cute invitation text that includes all the important details like date, time, location, RSVP information, and any special instructions about allergies or what kids should bring or wear",
            "The party theme colors should be pink, purple, and gold to match the unicorn decorations",
        ],
        user_variables={
            "SHOPPING_LIST": shopping_list.value,
            "PARTY_GAMES": party_games.value,
            "PARTY_TIMELINE": party_timeline.value,
        },
    )
    assert invitation_text.value is not None, (
        'ERROR: task "invitation_text" execution failed'
    )

    # 5. Develop a backup plan in case it rains, ensuring all activities can be moved indoors. - - BACKUP_PLAN
    backup_plan = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to develop a backup plan in case it rains, ensuring all activities can be moved indoors. Follow these steps to accomplish your task:

         First, review the shopping list, party games, timeline, and invitation text from the previous steps to understand the party setup and activities:
         <shopping_list>
         {{SHOPPING_LIST}}
         </shopping_list>
         <party_games>
         {{PARTY_GAMES}}
         </party_games>
         <party_timeline>
         {{PARTY_TIMELINE}}
         </party_timeline>
         <invitation_text>
         {{INVITATION_TEXT}}
         </invitation_text>

         Next, consider the following factors for the backup plan:
         1. **Indoor Space**: Ensure that the indoor space is large enough to accommodate all the activities and guests comfortably.
         2. **Activity Adjustments**: Modify the party games and activities to fit the indoor space. Consider the following:
            - **Space Constraints**: Ensure that games like "Pin the Horn on the Unicorn" or "Unicorn Ring Toss" can be played indoors without any hazards.
            - **Noise Levels**: Adjust activities to minimize noise, especially if the party is in a residential area.
            - **Safety**: Ensure that all indoor activities are safe and age-appropriate.
         3. **Food and Drinks**: Plan how to serve food and drinks indoors. Consider using tables or a buffet-style setup to minimize clutter.
         4. **Decorations**: Adjust the decoration setup to fit the indoor space. Ensure that the decorations are still visually appealing and fit the unicorn theme.
         5. **Timeline Adjustments**: Modify the party timeline to account for indoor activities. Ensure that the timeline is still efficient and keeps the kids entertained.

         Finally, write a detailed backup plan that includes:
         - **Indoor Activity Schedule**: A list of adjusted activities with their respective times.
         - **Indoor Setup Instructions**: Detailed instructions on how to set up the indoor space, including decorations, food, and drink stations.
         - **Safety Guidelines**: Any additional safety guidelines or considerations for indoor activities.

         Ensure that the backup plan is clear, concise, and easy to follow. The plan should allow for a smooth transition from outdoor to indoor activities in case of rain.
         """.strip()
        ),
        requirements=["Create a backup plan in case it rains"],
        user_variables={
            "SHOPPING_LIST": shopping_list.value,
            "PARTY_GAMES": party_games.value,
            "PARTY_TIMELINE": party_timeline.value,
            "INVITATION_TEXT": invitation_text.value,
        },
    )
    assert backup_plan.value is not None, 'ERROR: task "backup_plan" execution failed'

    # 6. Compile the shopping list, party games, timeline, invitation text, and backup plan into a single cohesive output. - - FINAL_PARTY_PLAN
    final_party_plan = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to compile the shopping list, party games, timeline, invitation text, and backup plan into a single cohesive output for the birthday party. Follow these steps to accomplish your task:

         1. **Review the Shopping List**:
            Carefully review the detailed shopping list with estimated costs for decorations, food, drinks, party favors, and a cake that serves at least 20 people. Ensure the total budget stays within $400.
            <shopping_list>
            {{SHOPPING_LIST}}
            </shopping_list>

         2. **Review the Party Games**:
            Review the suggested 5-7 age-appropriate party games or activities that fit the unicorn theme and can keep the kids entertained for about 3 hours.
            <party_games>
            {{PARTY_GAMES}}
            </party_games>

         3. **Review the Party Timeline**:
            Review the timeline for the party day starting from 2 hours before guests arrive until cleanup is done, including food prep, decoration setup, and scheduling of activities.
            <party_timeline>
            {{PARTY_TIMELINE}}
            </party_timeline>

         4. **Review the Invitation Text**:
            Review the cute invitation text that includes all the important details like date, time, location, RSVP information, and any special instructions about allergies or what kids should bring or wear.
            <invitation_text>
            {{INVITATION_TEXT}}
            </invitation_text>

         5. **Review the Backup Plan**:
            Review the backup plan in case it rains, ensuring all activities can be moved indoors.
            <backup_plan>
            {{BACKUP_PLAN}}
            </backup_plan>

         6. **Compile the Information**:
            Combine all the reviewed information into a single cohesive output. Ensure that the output is well-organized and easy to follow. Include the following sections in your final output:
            - **Shopping List**: Detailed list with estimated costs.
            - **Party Games**: Suggested activities with descriptions.
            - **Party Timeline**: Step-by-step schedule for the party day.
            - **Invitation Text**: Cute and informative text for parents.
            - **Backup Plan**: Plan for indoor activities in case of rain.

         7. **Final Output**:
            Present the compiled information in a clear and concise manner. Ensure that all details are included and that the output is easy to read and understand.

         Your final output should be a comprehensive guide that includes all the necessary information for planning and executing the birthday party.
         """.strip()
        ),
        requirements=[
            "Everything must stay within a total budget of about ~$400",
            "Follow basic food safety guidelines especially for kids with common allergies like nuts and dairy",
            "The party theme colors should be pink, purple, and gold to match the unicorn decorations",
        ],
        user_variables={
            "SHOPPING_LIST": shopping_list.value,
            "PARTY_GAMES": party_games.value,
            "PARTY_TIMELINE": party_timeline.value,
            "INVITATION_TEXT": invitation_text.value,
            "BACKUP_PLAN": backup_plan.value,
        },
    )
    assert final_party_plan.value is not None, (
        'ERROR: task "final_party_plan" execution failed'
    )

    final_answer = final_party_plan.value

    print(final_answer)
