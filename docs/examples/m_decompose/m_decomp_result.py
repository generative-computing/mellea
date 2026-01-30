# pytest: ollama, requires_heavy_ram, llm

import textwrap

import mellea

if __name__ == "__main__":
    m = mellea.start_session()

    # 1. Gather all necessary information about the company, event requirements, and constraints. - - INFORMATION_GATHERING
    information_gathering = m.instruct(
        textwrap.dedent(
            R"""
         To gather all necessary information about the company, event requirements, and constraints, follow these steps:

         1. **Company Information**:
            - **Company Size**: Determine the number of employees (100-300).
            - **Departments**: Identify the different departments within the company.
            - **Employee Demographics**: Gather information about the diversity of employees, including age, seniority, and roles.
            - **Company Culture**: Understand the company's values, mission, and current team dynamics.

         2. **Event Requirements**:
            - **Objectives**: Clearly define the goals of the event, such as strengthening team cohesion, improving cross-departmental collaboration, boosting morale, and creating lasting memories.
            - **Event Type**: Specify the type of event (e.g., corporate team-building event).
            - **Duration**: Determine the start and end times of the event (8:00 AM to 8:00 PM).
            - **Key Activities**: List the key activities planned for the event, such as registration, breakfast, opening ceremony, icebreaker activities, workshops, lunch, team challenges, awards ceremony, and dinner with entertainment.

         3. **Event Constraints**:
            - **Budget**: Identify the budget constraints for different aspects of the event, including venue rental ($15,000-25,000), catering ($75-125 per person), entertainment and activities ($10,000-20,000), transportation ($3,000-8,000), technology ($5,000-10,000), and total estimated budget ($50,000-125,000).
            - **Venue Requirements**: Specify the venue requirements, such as capacity for 100-300 people, accessibility, parking, indoor/outdoor space, AV capabilities, WiFi for 300+ devices, catering facilities, accommodations, and cost-effectiveness.
            - **Safety and Accessibility**: Ensure the event follows best practices for safety, accessibility, and sustainability.
            - **Technology**: Identify the technology requirements, such as event app for scheduling and messaging, AV equipment, WiFi for 300+ connections, photography services, live streaming, and tech support.

         4. **Additional Information**:
            - **Marketing and Communication**: Gather information about the marketing and communication strategy, including save-the-date announcements, formal invitations, reminder emails, internal campaigns, dedicated Slack channel, executive videos, and mobile event app.
            - **Team Formation**: Understand the team formation strategy, including diverse mix of departments and seniority, pre-event survey for dietary restrictions and preferences, and team assignments 2 weeks before the event.
            - **Workshop Content**: Identify the content for leadership, innovation, communication, and wellness workshops.
            - **Catering Plan**: Gather information about the catering plan, including breakfast, lunch, afternoon snacks, and dinner options.
            - **Activity Logistics**: Understand the logistics for all planned activities, including equipment lists, setup instructions, safety protocols, facilitator guides, scoring systems, and rotation schedules.
            - **Transportation Plan**: Gather information about the transportation plan, including charter buses, parking management, shuttle service, and contingency plans.
            - **Registration Process**: Understand the registration process, including online portal, on-site check-in, name tags, welcome packets, and staff training.
            - **Safety Plan**: Gather information about the safety plan, including insurance, first aid stations, evacuation procedures, weather contingencies, food safety, security personnel, and incident reporting.
            - **Vendor Management**: Understand the vendor management strategy, including selection criteria, contracts, payment schedules, and evaluation forms.
            - **Technology Plan**: Gather information about the technology plan, including event app, AV equipment, WiFi, photography services, live streaming, and tech support.
            - **Decoration Strategy**: Understand the decoration strategy, including signage, team colors, photo backdrops, and sustainable options.
            - **Entertainment Plan**: Gather information about the entertainment plan, including live band or DJ, emcee, interactive activities, and evening games.
            - **Prize Strategy**: Understand the prize strategy, including team prizes, individual awards, participation gifts, and raffle prizes.
            - **Post-Event Evaluation**: Gather information about the post-event evaluation plan, including participant survey, debriefing sessions, budget reconciliation, photo compilation, impact assessment, and thank you communications.
            - **Accessibility Plan**: Understand the accessibility plan, including ADA compliance, interpreters, activity modifications, dietary accommodation, and inclusive team formation.
            - **Sustainability Plan**: Gather information about the sustainability plan, including eco-friendly venue, local catering, reusable serving ware, digital communications, recyclable decorations, and waste management.
            - **Contingency Planning**: Understand the contingency planning for weather emergencies, vendor cancellations, technology failures, medical emergencies, and budget overruns.

         5. **Documentation**:
            - Compile all the gathered information into a comprehensive document that will serve as the basis for the event strategy document.

         By following these steps, you will gather all necessary information about the company, event requirements, and constraints to ensure a successful and well-organized corporate team-building event.
         """.strip()
        ),
        requirements=[
            "The budget breakdowns must include venue rental at $15,000-25,000, catering at $75-125 per person, entertainment and activities at $10,000-20,000, transportation at $3,000-8,000, technology at $5,000-10,000, and total estimated budget of $50,000-125,000",
            "The venue selection process must evaluate 10 venues based on capacity for 100-300 people, accessibility, parking, indoor/outdoor space, AV capabilities, WiFi for 300+ devices, catering facilities, accommodations, and cost-effectiveness",
            "The safety plan must include insurance, first aid stations, evacuation procedures, weather contingencies, food safety with allergen labeling, security personnel, and incident reporting",
            "The accessibility plan must include ADA compliance, interpreters, activity modifications, dietary accommodation, and inclusive team formation",
            "The sustainability plan must include eco-friendly venue, local catering, reusable serving ware, digital communications, recyclable decorations, and waste management",
            "All planning must follow best practices, create memorable experiences, accommodate diverse needs, stay within budget, prioritize safety, incorporate sustainable practices, leverage technology, build excitement, and measure success",
        ],
    )
    assert information_gathering.value is not None, (
        'ERROR: task "information_gathering" execution failed'
    )

    # 2. Create a detailed planning timeline starting 6 months before the event with weekly milestones. - - PLANNING_TIMELINE
    planning_timeline = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to create a detailed planning timeline starting 6 months before the event with weekly milestones. Follow these steps to accomplish your task:

         First, review the gathered information about the company, event requirements, and constraints from the previous step:
         <gathered_information>
         {{INFORMATION_GATHERING}}
         </gathered_information>

         Next, consider the key components of the event that need to be planned, such as venue selection, marketing and communication, team formation, workshops, catering, activities, transportation, registration, safety, vendor management, technology, decorations, entertainment, prizes, post-event evaluation, accessibility, sustainability, and contingency planning.

         Break down the planning process into weekly milestones starting 6 months before the event. Ensure that each milestone is specific, measurable, achievable, relevant, and time-bound (SMART).

         Here is a suggested structure for the planning timeline:

         **6 Months Before the Event:**
         - Finalize event objectives and key performance indicators (KPIs).
         - Establish the event budget and allocate funds to different categories.
         - Develop a high-level event plan and timeline.
         - Identify and assemble the event planning team.
         - Begin researching and evaluating potential venues.

         **5 Months Before the Event:**
         - Select and book the venue.
         - Develop a detailed event schedule and share it with key stakeholders.
         - Create a marketing and communication strategy.
         - Begin designing the event branding and promotional materials.
         - Start planning the team formation strategy.

         **4 Months Before the Event:**
         - Send save-the-date announcements to employees.
         - Finalize the event schedule and share it with vendors and participants.
         - Develop the workshop content and identify facilitators.
         - Create a catering plan and send out a pre-event survey for dietary restrictions.
         - Begin planning the icebreaker activities.

         **3 Months Before the Event:**
         - Send formal invitations to employees.
         - Finalize the team formation strategy and assign teams.
         - Develop the activity logistics and equipment lists.
         - Create a transportation plan and arrange charter buses if necessary.
         - Design the registration process and set up the online portal.

         **2 Months Before the Event:**
         - Send weekly reminder emails to employees.
         - Finalize the catering plan and confirm menu options.
         - Develop the safety plan and evacuation procedures.
         - Create a vendor management strategy and finalize contracts.
         - Design the technology plan and arrange AV equipment.

         **1 Month Before the Event:**
         - Send team assignments and welcome packets to employees.
         - Finalize the decoration strategy and order necessary materials.
         - Develop the entertainment plan and book a live band or DJ.
         - Create a prize strategy and order prizes.
         - Finalize the post-event evaluation plan.

         **2 Weeks Before the Event:**
         - Conduct a final walkthrough of the venue.
         - Train staff on the registration process and safety protocols.
         - Finalize the accessibility plan and make necessary accommodations.
         - Develop a sustainability plan and implement eco-friendly practices.
         - Create a contingency plan for potential emergencies.

         **1 Week Before the Event:**
         - Send a final reminder email to employees.
         - Confirm all vendor arrangements and payment schedules.
         - Finalize the event schedule and share it with participants.
         - Prepare the welcome packets and name tags.
         - Conduct a final review of the event strategy document.

         Ensure that the planning timeline is comprehensive and covers all aspects of the event. The timeline should be flexible and allow for adjustments as needed.

         Finally, write the detailed planning timeline starting 6 months before the event with weekly milestones. Your answer will serve as basis for the next steps.
         """.strip()
        ),
        requirements=[
            "The event strategy document must include detailed planning timelines starting 6 months before the event with weekly milestones"
        ],
        user_variables={"INFORMATION_GATHERING": information_gathering.value},
    )
    assert planning_timeline.value is not None, (
        'ERROR: task "planning_timeline" execution failed'
    )

    # 3. Develop a budget breakdown for venue rental, catering, entertainment, transportation, technology, and total estimated budget. - - BUDGET_BREAKDOWN
    budget_breakdown = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to develop a detailed budget breakdown for the corporate team-building event. This budget should include estimates for venue rental, catering, entertainment, transportation, technology, and the total estimated budget.

         To accomplish this, follow these steps:

         1. **Review Gathered Information**:
            First, review the information gathered about the company, event requirements, and constraints. This information will help you understand the scope and specific needs of the event:
            <gathered_information>
            {{INFORMATION_GATHERING}}
            </gathered_information>

         2. **Understand Budget Constraints**:
            The event has specific budget constraints for each category:
            - **Venue Rental**: $15,000 - $25,000
            - **Catering**: $75 - $125 per person
            - **Entertainment and Activities**: $10,000 - $20,000
            - **Transportation**: $3,000 - $8,000
            - **Technology**: $5,000 - $10,000
            - **Total Estimated Budget**: $50,000 - $125,000

         3. **Calculate Catering Costs**:
            The catering cost is per person. The event will have 100-300 employees attending. Calculate the total catering cost based on the number of attendees and the per-person cost.

         4. **Estimate Other Costs**:
            Provide a detailed estimate for each of the remaining categories (venue rental, entertainment, transportation, and technology) within the specified budget ranges.

         5. **Summarize Total Estimated Budget**:
            Add up the estimated costs for each category to provide a total estimated budget for the event. Ensure that the total estimated budget falls within the specified range of $50,000 - $125,000.

         6. **Provide Detailed Breakdown**:
            Present the budget breakdown in a clear and organized manner. Include the following information for each category:
            - **Category**: Name of the budget category (e.g., Venue Rental)
            - **Estimated Cost**: The estimated cost for that category
            - **Notes**: Any additional notes or considerations for that category

         7. **Review and Adjust**:
            Review your budget breakdown to ensure it is accurate and aligns with the event's requirements and constraints. Make any necessary adjustments to ensure the total estimated budget is within the specified range.

         8. **Final Answer**:
            Provide the detailed budget breakdown as your final answer. Ensure it is clear, concise, and adheres to the specified budget constraints.

         Example Format:
         - **Venue Rental**: $20,000
            - Notes: Includes rental fee for the venue and basic setup.
         - **Catering**: $25,000
            - Notes: Based on 200 attendees at $125 per person.
         - **Entertainment and Activities**: $15,000
            - Notes: Includes costs for activities and entertainment.
         - **Transportation**: $5,000
            - Notes: Includes charter buses and shuttle service.
         - **Technology**: $7,000
            - Notes: Includes AV equipment, WiFi, and tech support.
         - **Total Estimated Budget**: $72,000
         """.strip()
        ),
        requirements=[
            "The budget breakdowns must include venue rental at $15,000-25,000, catering at $75-125 per person, entertainment and activities at $10,000-20,000, transportation at $3,000-8,000, technology at $5,000-10,000, and total estimated budget of $50,000-125,000"
        ],
        user_variables={"INFORMATION_GATHERING": information_gathering.value},
    )
    assert budget_breakdown.value is not None, (
        'ERROR: task "budget_breakdown" execution failed'
    )

    # 4. Design a marketing and communication strategy including save-the-date announcements, formal invitations, reminder emails, internal campaigns, and other communication methods. - - MARKETING_STRATEGY
    marketing_strategy = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to design a comprehensive marketing and communication strategy for a corporate team-building event. This strategy should ensure effective engagement and participation from all employees. Follow these steps to accomplish your task:

         1. **Review Gathered Information**:
            Begin by reviewing the information gathered about the company, event requirements, and constraints. This will help you tailor the marketing and communication strategy to the specific needs and preferences of the employees:
            <information_gathering>
            {{INFORMATION_GATHERING}}
            </information_gathering>

         2. **Understand the Planning Timeline**:
            Familiarize yourself with the planning timeline to ensure that all marketing and communication activities are scheduled appropriately. This will help you create a timeline for save-the-date announcements, formal invitations, reminder emails, and internal campaigns:
            <planning_timeline>
            {{PLANNING_TIMELINE}}
            </planning_timeline>

         3. **Budget Considerations**:
            Consider the budget breakdown to ensure that the marketing and communication strategy is cost-effective and aligns with the overall budget for the event:
            <budget_breakdown>
            {{BUDGET_BREAKDOWN}}
            </budget_breakdown>

         4. **Develop the Marketing and Communication Strategy**:
            Create a detailed marketing and communication strategy that includes the following components:

            - **Save-the-Date Announcements**:
               Plan to send save-the-date announcements 4 months in advance. These announcements should be sent via email and should include the event date, purpose, and a brief overview of what to expect.

            - **Formal Invitations**:
               Design formal invitations to be sent 2 months before the event. These invitations should include detailed information about the event schedule, dress code, and any specific instructions or requirements.

            - **Weekly Reminder Emails**:
               Schedule weekly reminder emails leading up to the event. These emails should provide updates, reminders, and any additional information that may be relevant to the participants.

            - **Internal Campaigns**:
               Develop internal campaigns to build excitement and engagement. These campaigns can include posters, flyers, and announcements on internal communication platforms.

            - **Dedicated Slack Channel**:
               Create a dedicated Slack channel for event-related discussions, updates, and announcements. This channel will serve as a central hub for all event-related communication.

            - **Executive Videos**:
               Plan to create and share videos featuring executive messages about the event. These videos can be shared via email, internal communication platforms, and the dedicated Slack channel.

            - **Mobile Event App**:
               Develop a mobile event app that provides participants with access to the event schedule, maps, speaker information, and other relevant details. The app should also include features for messaging and networking.

         5. **Ensure Consistency and Clarity**:
            Ensure that all marketing and communication materials are consistent in tone, style, and branding. This will help create a cohesive and professional image for the event.

         6. **Review and Finalize**:
            Review the marketing and communication strategy to ensure that it meets all the requirements and aligns with the overall event goals. Make any necessary adjustments and finalize the strategy.

         Your final answer should be a detailed marketing and communication strategy document that includes all the components mentioned above.
         """.strip()
        ),
        requirements=[
            "The marketing and communication strategy must include save-the-date announcements 4 months in advance, formal invitations 2 months before, weekly reminder emails, internal campaigns, dedicated Slack channel, executive videos, and mobile event app"
        ],
        user_variables={
            "INFORMATION_GATHERING": information_gathering.value,
            "PLANNING_TIMELINE": planning_timeline.value,
            "BUDGET_BREAKDOWN": budget_breakdown.value,
        },
    )
    assert marketing_strategy.value is not None, (
        'ERROR: task "marketing_strategy" execution failed'
    )

    # 5. Develop a venue selection process to evaluate and select the best venue based on capacity, accessibility, parking, indoor/outdoor space, AV capabilities, WiFi, catering facilities, accommodations, and cost-effectiveness. - - VENUE_SELECTION
    venue_selection = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to develop a comprehensive venue selection process to evaluate and select the best venue for the corporate team-building event. Follow these steps to accomplish your task:

         1. **Understand the Requirements**:
            Review the gathered information about the company and event requirements, including the need to accommodate 100-300 employees. Ensure the venue selection process aligns with the overall event goals of strengthening team cohesion, improving cross-departmental collaboration, boosting morale, and creating lasting memories.

         2. **Define Evaluation Criteria**:
            Based on the event requirements, define the key criteria for evaluating potential venues. The criteria should include:
            - **Capacity**: Ability to accommodate 100-300 people.
            - **Accessibility**: Ease of access for all attendees, including those with disabilities.
            - **Parking**: Adequate parking facilities for attendees and staff.
            - **Indoor/Outdoor Space**: Availability of both indoor and outdoor spaces for various activities.
            - **AV Capabilities**: Sufficient audio-visual equipment and support.
            - **WiFi**: Reliable WiFi for 300+ devices.
            - **Catering Facilities**: Ability to support breakfast, lunch, afternoon snacks, and dinner.
            - **Accommodations**: Proximity to accommodations for out-of-town attendees, if necessary.
            - **Cost-Effectiveness**: Budget-friendly options within the estimated budget of $15,000-25,000.

         3. **Research Potential Venues**:
            Identify at least 10 potential venues that meet the initial criteria. Gather detailed information about each venue, including:
            - Location and contact information.
            - Capacity and layout.
            - Availability of required facilities and services.
            - Cost estimates for rental and additional services.

         4. **Create Evaluation Forms**:
            Develop evaluation forms to systematically assess each venue against the defined criteria. The forms should include:
            - Checklists for each criterion.
            - Rating scales for subjective assessments.
            - Space for notes and additional comments.

         5. **Conduct Venue Visits or Virtual Tours**:
            Schedule visits or virtual tours of the shortlisted venues. During the visits, evaluate:
            - The overall ambiance and suitability for the event.
            - The condition and functionality of facilities.
            - The responsiveness and professionalism of the venue staff.

         6. **Compare and Contrast Venues**:
            Use the evaluation forms to compare and contrast the venues. Create a summary table or matrix that highlights the strengths and weaknesses of each venue based on the defined criteria.

         7. **Select the Best Venue**:
            Based on the evaluations, select the venue that best meets the event requirements and aligns with the budget. Consider factors such as overall value, flexibility, and the ability to create a memorable experience for attendees.

         8. **Document the Selection Process**:
            Prepare a detailed report documenting the venue selection process. The report should include:
            - The defined evaluation criteria.
            - The list of potential venues and their evaluations.
            - The comparison and contrast of venues.
            - The rationale for selecting the final venue.

         9. **Finalize Venue Contract**:
            Work with the selected venue to finalize the contract, ensuring all agreed-upon terms and conditions are clearly outlined. Include details such as:
            - Rental costs and payment schedules.
            - Cancellation policies.
            - Additional services and their costs.

         10. **Integrate with Overall Event Plan**:
               Ensure the selected venue aligns with the overall event strategy document. Coordinate with other event planning components, such as catering, transportation, and technology, to ensure seamless integration.

         By following these steps, you will develop a thorough and effective venue selection process that meets the event's requirements and contributes to its success.
         """.strip()
        ),
        requirements=[
            "The venue selection process must evaluate 10 venues based on capacity for 100-300 people, accessibility, parking, indoor/outdoor space, AV capabilities, WiFi for 300+ devices, catering facilities, accommodations, and cost-effectiveness"
        ],
        user_variables={},
    )
    assert venue_selection.value is not None, (
        'ERROR: task "venue_selection" execution failed'
    )

    # 6. Design a detailed event schedule from 8:00 AM to 8:00 PM including registration, breakfast, opening ceremony, icebreaker activities, workshops, lunch, team challenges, awards ceremony, and dinner with entertainment. - - EVENT_SCHEDULE
    event_schedule = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to design a detailed event schedule from 8:00 AM to 8:00 PM for a corporate team-building event. The schedule should include the following key components:

         1. **Registration and Breakfast (8:00 - 9:00 AM)**
            - Plan for registration process and breakfast setup.
            - Include details on coffee stations, pastries, and any other breakfast items.

         2. **Opening Ceremony (9:00 - 9:30 AM)**
            - Outline the opening ceremony, including the CEO's speech.
            - Include any additional opening activities or announcements.

         3. **Icebreaker Activities (9:30 - 10:30 AM)**
            - Plan for icebreaker activities such as human bingo and speed networking.
            - Include any necessary setup or facilitation details.

         4. **Morning Workshops (10:30 AM - 12:30 PM)**
            - Design tracks on leadership, innovation, communication, and wellness.
            - Include details on workshop content, facilitators, and any required materials.

         5. **Lunch (12:30 - 2:00 PM)**
            - Plan for lunch with diverse menu options.
            - Include details on catering setup, dietary restrictions, and any special considerations.

         6. **Afternoon Team Challenges (2:00 - 5:00 PM)**
            - Plan for team challenges such as scavenger hunt, escape rooms, outdoor adventures, cooking competition, and sports tournaments.
            - Include details on equipment lists, setup instructions, safety protocols, facilitator guides, scoring systems, and rotation schedules.

         7. **Awards Ceremony (5:00 - 5:30 PM)**
            - Plan for the awards ceremony, including any necessary setup or facilitation details.
            - Include details on award categories, presentation, and any additional activities.

         8. **Dinner with Entertainment (5:30 - 8:00 PM)**
            - Plan for dinner with three entree choices and a full bar.
            - Include details on entertainment such as live band or DJ, emcee, interactive activities, and evening games.

         To accomplish this task, follow these steps:

         1. **Review Gathered Information**:
            - Carefully review the gathered information about the company, event requirements, and constraints from the previous step:
            <information_gathering>
            {{INFORMATION_GATHERING}}
            </information_gathering>

         2. **Review Planning Timeline**:
            - Review the planning timeline starting 6 months before the event with weekly milestones from the previous step:
            <planning_timeline>
            {{PLANNING_TIMELINE}}
            </planning_timeline>

         3. **Review Budget Breakdown**:
            - Review the budget breakdown for venue rental, catering, entertainment, transportation, technology, and total estimated budget from the previous step:
            <budget_breakdown>
            {{BUDGET_BREAKDOWN}}
            </budget_breakdown>

         4. **Review Marketing Strategy**:
            - Review the marketing and communication strategy including save-the-date announcements, formal invitations, reminder emails, internal campaigns, and other communication methods from the previous step:
            <marketing_strategy>
            {{MARKETING_STRATEGY}}
            </marketing_strategy>

         5. **Review Venue Selection**:
            - Review the venue selection process to evaluate and select the best venue based on capacity, accessibility, parking, indoor/outdoor space, AV capabilities, WiFi, catering facilities, accommodations, and cost-effectiveness from the previous step:
            <venue_selection>
            {{VENUE_SELECTION}}
            </venue_selection>

         6. **Design the Event Schedule**:
            - Use the information gathered in the previous steps to design a detailed event schedule from 8:00 AM to 8:00 PM.
            - Ensure the schedule includes all the key components mentioned above.
            - Make sure the schedule is realistic, feasible, and aligns with the event's goals and constraints.

         7. **Finalize the Schedule**:
            - Review the designed event schedule to ensure it meets all the requirements and constraints.
            - Make any necessary adjustments to ensure the schedule is comprehensive and detailed.

         8. **Output the Event Schedule**:
            - Provide the detailed event schedule as your final answer. Ensure the output is clear, concise, and well-organized.

         Remember to adhere to the event's goals of strengthening team cohesion, improving cross-departmental collaboration, boosting morale, and creating lasting memories. Ensure the schedule is inclusive, accessible, and sustainable, following best practices and leveraging technology where appropriate.
         """.strip()
        ),
        requirements=[
            "The event schedule must include registration and breakfast at 8:00-9:00 AM, opening ceremony at 9:00-9:30 AM with CEO speech, icebreaker activities at 9:30-10:30 AM including human bingo and speed networking, morning workshops at 10:30 AM-12:30 PM with tracks on leadership, innovation, communication, and wellness, lunch at 12:30-2:00 PM with diverse menu options, afternoon team challenges at 2:00-5:00 PM including scavenger hunt, escape rooms, outdoor adventures, cooking competition, and sports tournaments, awards ceremony at 5:00-5:30 PM, and dinner with entertainment at 5:30-8:00 PM"
        ],
        user_variables={
            "INFORMATION_GATHERING": information_gathering.value,
            "PLANNING_TIMELINE": planning_timeline.value,
            "BUDGET_BREAKDOWN": budget_breakdown.value,
            "MARKETING_STRATEGY": marketing_strategy.value,
            "VENUE_SELECTION": venue_selection.value,
        },
    )
    assert event_schedule.value is not None, (
        'ERROR: task "event_schedule" execution failed'
    )

    # 7. Create a list of 15 icebreaker activities with detailed plans. - - ICEBREAKER_PLANS
    icebreaker_plans = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to create a list of 15 icebreaker activities with detailed plans for a corporate team-building event. These activities should be engaging, inclusive, and designed to foster team cohesion and collaboration among 100-300 employees.

         To approach this task, follow these steps:

         1. **Understand the Event Context**:
            Review the gathered information about the company, event requirements, and constraints from the previous step:
            <information_gathering>
            {{INFORMATION_GATHERING}}
            </information_gathering>

         2. **Review the Event Schedule**:
            Ensure that the icebreaker activities align with the event schedule, particularly the time slot allocated for icebreaker activities from 9:30 AM to 10:30 AM:
            <event_schedule>
            {{EVENT_SCHEDULE}}
            </event_schedule>

         3. **List of Icebreaker Activities**:
            Create a list of 15 icebreaker activities. Each activity should include the following details:
            - **Activity Name**: A clear and descriptive title.
            - **Objective**: The purpose of the activity and how it contributes to team-building.
            - **Materials Needed**: A list of any equipment or materials required.
            - **Setup Instructions**: Step-by-step instructions for setting up the activity.
            - **Instructions for Participants**: Clear guidelines on how to participate.
            - **Duration**: The estimated time required for the activity.
            - **Facilitator Notes**: Any special instructions or tips for the facilitator.
            - **Safety Protocols**: Any safety considerations or guidelines.
            - **Scoring System (if applicable)**: How the activity will be scored or evaluated.
            - **Rotation Schedule (if applicable)**: How the activity will be rotated among teams.

         4. **Diverse and Inclusive Activities**:
            Ensure that the activities are diverse and inclusive, catering to different personalities, abilities, and preferences. Consider activities that can be adapted for various team sizes and dynamics.

         5. **Engaging and Fun**:
            The activities should be engaging and fun, designed to create a positive and memorable experience for all participants.

         6. **Alignment with Event Goals**:
            Ensure that the activities align with the overall goals of the event, which include strengthening team cohesion, improving cross-departmental collaboration, boosting morale, and creating lasting memories.

         Here is an example structure to guide your writing:
         - **Activity 1: Human Knot**
            - **Objective**: Promote teamwork and problem-solving.
            - **Materials Needed**: None.
            - **Setup Instructions**: Have participants stand in a circle.
            - **Instructions for Participants**: Each person must hold hands with two people across from them to form a "human knot."
            - **Duration**: 15-20 minutes.
            - **Facilitator Notes**: Encourage communication and teamwork.
            - **Safety Protocols**: Ensure participants are careful not to pull too hard.
            - **Scoring System**: Not applicable.
            - **Rotation Schedule**: Not applicable.

         Repeat the above structure for all 15 icebreaker activities.

         Ensure that each activity is clearly described and includes all the necessary details for successful execution.

         Finally, compile the list of 15 icebreaker activities with detailed plans into a comprehensive document.
         """.strip()
        ),
        requirements=[
            "The icebreaker plans must include 15 options including human knot, marshmallow tower, obstacle course, pictionary, charades, and trust falls",
            "The event schedule must include registration and breakfast at 8:00-9:00 AM, opening ceremony at 9:00-9:30 AM with CEO speech, icebreaker activities at 9:30-10:30 AM including human bingo and speed networking, morning workshops at 10:30 AM-12:30 PM with tracks on leadership, innovation, communication, and wellness, lunch at 12:30-2:00 PM with diverse menu options, afternoon team challenges at 2:00-5:00 PM including scavenger hunt, escape rooms, outdoor adventures, cooking competition, and sports tournaments, awards ceremony at 5:00-5:30 PM, and dinner with entertainment at 5:30-8:00 PM",
            "The activity logistics must include equipment lists, setup instructions, safety protocols, facilitator guides, scoring systems, and rotation schedules",
            "The safety plan must include insurance, first aid stations, evacuation procedures, weather contingencies, food safety with allergen labeling, security personnel, and incident reporting",
            "The accessibility plan must include ADA compliance, interpreters, activity modifications, dietary accommodation, and inclusive team formation",
        ],
        user_variables={
            "INFORMATION_GATHERING": information_gathering.value,
            "EVENT_SCHEDULE": event_schedule.value,
        },
    )
    assert icebreaker_plans.value is not None, (
        'ERROR: task "icebreaker_plans" execution failed'
    )

    # 8. Develop a team formation strategy with diverse mix of departments and seniority, pre-event survey for dietary restrictions, and team assignments 2 weeks before the event. - - TEAM_FORMATION
    team_formation = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to develop a team formation strategy for a corporate team-building event. This strategy should ensure a diverse mix of departments and seniority levels, accommodate dietary restrictions, and assign teams two weeks before the event.

         To accomplish this, follow these steps:

         1. **Understand the Event Requirements**:
            Review the gathered information about the company, event requirements, and constraints from the previous step:
            <information_gathering>
            {{INFORMATION_GATHERING}}
            </information_gathering>

         2. **Diverse Team Formation**:
            Create a strategy to form teams with a diverse mix of departments and seniority levels. Each team should consist of 8-12 people. Ensure that the teams are balanced in terms of experience, roles, and departments to foster collaboration and learning.

         3. **Pre-Event Survey**:
            Design a pre-event survey to collect information about dietary restrictions and preferences from all participants. This survey should be distributed at least one month before the event to allow sufficient time for planning.

         4. **Team Assignments**:
            Develop a process for assigning participants to teams. This process should consider the survey responses regarding dietary restrictions and ensure that teams are formed in a way that accommodates these needs. Team assignments should be communicated to participants two weeks before the event.

         5. **Team Identification**:
            Assign unique names and colors to each team to facilitate easy identification and foster team spirit. Ensure that the names and colors are inclusive and appropriate for all participants.

         6. **Communication Plan**:
            Outline a communication plan for informing participants about their team assignments, including the team name, color, and any other relevant information. This plan should include the distribution of team assignments and any follow-up communications.

         7. **Documentation**:
            Document the team formation strategy, including the criteria used for team formation, the survey questions, the process for assigning teams, and the communication plan. This documentation will be included in the comprehensive event strategy document.

         Ensure that your strategy aligns with the overall event goals of strengthening team cohesion, improving cross-departmental collaboration, boosting morale, and creating lasting memories.
         """.strip()
        ),
        requirements=[
            "The team formation strategy must include diverse mix of departments and seniority in teams of 8-12 people, pre-event survey for dietary restrictions and preferences, and team assignments 2 weeks before with names and colors"
        ],
        user_variables={"INFORMATION_GATHERING": information_gathering.value},
    )
    assert team_formation.value is not None, (
        'ERROR: task "team_formation" execution failed'
    )

    # 9. Design content for leadership, innovation, communication, and wellness workshops. - - WORKSHOP_CONTENT
    workshop_content = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to design content for leadership, innovation, communication, and wellness workshops as part of the corporate team-building event. Follow these steps to accomplish your task:

         1. **Understand the Workshop Requirements**:
            - The workshops should focus on four key areas: leadership, innovation, communication, and wellness.
            - Each workshop should be engaging, informative, and relevant to the attendees, who are employees from various departments and seniority levels.
            - The workshops should align with the overall goals of strengthening team cohesion, improving cross-departmental collaboration, boosting morale, and creating lasting memories.

         2. **Review Gathered Information**:
            - Use the information gathered in the previous steps to understand the company's needs, event requirements, and constraints:
               <information_gathering>
               {{INFORMATION_GATHERING}}
               </information_gathering>

         3. **Design Leadership Workshop Content**:
            - Create a workshop that focuses on leadership case studies, practical exercises, and group discussions.
            - Include real-world examples and interactive activities that encourage participants to apply leadership principles.
            - Ensure the content is relevant to both current leaders and those aspiring to leadership roles.

         4. **Design Innovation Workshop Content**:
            - Develop a workshop that emphasizes innovation brainstorming, creative problem-solving, and idea generation.
            - Incorporate activities that encourage out-of-the-box thinking and collaboration.
            - Provide tools and techniques that participants can use to foster innovation within their teams.

         5. **Design Communication Workshop Content**:
            - Create a workshop that covers effective communication strategies, active listening, and conflict resolution.
            - Include exercises that help participants practice clear and concise communication.
            - Address both verbal and non-verbal communication skills.

         6. **Design Wellness Workshop Content**:
            - Develop a workshop that focuses on wellness meditation, stress management, and work-life balance.
            - Incorporate activities that promote physical and mental well-being.
            - Provide practical tips and techniques that participants can integrate into their daily routines.

         7. **Ensure Alignment with Event Schedule**:
            - Refer to the detailed event schedule to ensure the workshops fit within the allocated time slots:
               <event_schedule>
               {{EVENT_SCHEDULE}}
               </event_schedule>

         8. **Consider Team Formation Strategy**:
            - Take into account the team formation strategy to ensure the workshops cater to diverse groups:
               <team_formation>
               {{TEAM_FORMATION}}
               </team_formation>

         9. **Create Detailed Workshop Plans**:
            - For each workshop, provide a detailed plan that includes:
               - Objectives and learning outcomes
               - Agenda and timeline
               - Materials and resources needed
               - Facilitator guidelines
               - Participant activities and exercises
               - Evaluation methods

         10. **Review and Finalize**:
               - Ensure the workshop content aligns with the overall event goals and constraints.
               - Make any necessary adjustments to ensure the workshops are engaging, informative, and relevant to the attendees.

         Your final answer should be a detailed plan for each of the four workshops, including all the elements listed above.
         """.strip()
        ),
        requirements=[
            "The workshop content must include leadership case studies, innovation brainstorming, communication exercises, and wellness meditation"
        ],
        user_variables={
            "INFORMATION_GATHERING": information_gathering.value,
            "EVENT_SCHEDULE": event_schedule.value,
            "TEAM_FORMATION": team_formation.value,
        },
    )
    assert workshop_content.value is not None, (
        'ERROR: task "workshop_content" execution failed'
    )

    # 10. Create a detailed catering plan with breakfast, lunch, afternoon snacks, and dinner options. - - CATERING_PLAN
    catering_plan = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to create a detailed catering plan for the corporate team-building event. The catering plan should include options for breakfast, lunch, afternoon snacks, and dinner, ensuring a diverse menu that accommodates various dietary restrictions and preferences.

         To approach this task, follow these steps:

         1. **Review Gathered Information**:
            Begin by reviewing the information gathered about the company and event requirements:
            <information_gathering>
            {{INFORMATION_GATHERING}}
            </information_gathering>

         2. **Understand the Event Schedule**:
            Familiarize yourself with the event schedule to ensure the catering plan aligns with the timing of meals and breaks:
            <event_schedule>
            {{EVENT_SCHEDULE}}
            </event_schedule>

         3. **Consider Dietary Restrictions**:
            Use the team formation strategy, which includes a pre-event survey for dietary restrictions and preferences, to ensure the catering plan accommodates all attendees:
            <team_formation>
            {{TEAM_FORMATION}}
            </team_formation>

         4. **Breakfast Options**:
            Plan a breakfast menu that includes a coffee station and pastries. Ensure there are options for different dietary needs, such as gluten-free, vegan, and lactose-free choices.

         5. **Lunch Options**:
            Design a lunch buffet with a variety of options, including a salad bar and hot entrees. Provide diverse menu choices to cater to different tastes and dietary restrictions.

         6. **Afternoon Snacks**:
            Plan for afternoon snacks that are light and refreshing. Consider options like fruits, vegetables, and healthy snacks to keep energy levels up during the event.

         7. **Dinner Options**:
            Create a dinner menu with three entree choices and a full bar. Ensure there are options for different dietary needs and preferences, such as vegetarian, vegan, and gluten-free choices.

         8. **Allergen Labeling**:
            Include allergen labeling for all food items to ensure the safety and well-being of all attendees.

         9. **Sustainability Considerations**:
            Incorporate sustainable practices into the catering plan, such as using reusable serving ware and minimizing food waste.

         10. **Budget Constraints**:
               Ensure the catering plan stays within the budget constraints outlined in the budget breakdown:
               <budget_breakdown>
               {{BUDGET_BREAKDOWN}}
               </budget_breakdown>

         11. **Finalize the Catering Plan**:
               Compile all the information into a detailed catering plan that includes timings, menu options, and any special considerations.

         Your final answer should be a comprehensive catering plan that meets all the requirements and constraints of the event.
         """.strip()
        ),
        requirements=[
            "The catering plan must include breakfast coffee station and pastries, lunch buffet with salad bar and hot entrees, afternoon snacks, and dinner with three entree choices and full bar",
            "The safety plan must include insurance, first aid stations, evacuation procedures, weather contingencies, food safety with allergen labeling, security personnel, and incident reporting",
            "The sustainability plan must include eco-friendly venue, local catering, reusable serving ware, digital communications, recyclable decorations, and waste management",
            "The team formation strategy must include diverse mix of departments and seniority in teams of 8-12 people, pre-event survey for dietary restrictions and preferences, and team assignments 2 weeks before with names and colors",
            "The budget breakdowns must include venue rental at $15,000-25,000, catering at $75-125 per person, entertainment and activities at $10,000-20,000, transportation at $3,000-8,000, technology at $5,000-10,000, and total estimated budget of $50,000-125,000",
        ],
        user_variables={
            "INFORMATION_GATHERING": information_gathering.value,
            "EVENT_SCHEDULE": event_schedule.value,
            "TEAM_FORMATION": team_formation.value,
            "BUDGET_BREAKDOWN": budget_breakdown.value,
        },
    )
    assert catering_plan.value is not None, (
        'ERROR: task "catering_plan" execution failed'
    )

    # 11. Develop logistics for all planned activities including equipment lists, setup instructions, safety protocols, facilitator guides, scoring systems, and rotation schedules. - - ACTIVITY_LOGISTICS
    activity_logistics = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to develop detailed logistics for all planned activities for the corporate team-building event. This includes creating equipment lists, setup instructions, safety protocols, facilitator guides, scoring systems, and rotation schedules.

         To approach this task, follow these steps:

         1. **Review the Event Schedule and Activity Plans**:
            - Carefully review the event schedule and the list of planned activities from the previous steps. These include icebreaker activities, workshops, team challenges, and other events.
            - Ensure you have a clear understanding of the timeline and the sequence of activities.

         2. **Create Equipment Lists**:
            - For each activity, identify the necessary equipment and materials required.
            - Ensure that the equipment lists are comprehensive and include all items needed for smooth execution.

         3. **Develop Setup Instructions**:
            - Provide detailed setup instructions for each activity.
            - Include information on how to arrange the space, set up equipment, and prepare the area for participants.

         4. **Establish Safety Protocols**:
            - Develop safety protocols for each activity to ensure the well-being of all participants.
            - Include guidelines for emergency procedures, first aid, and any specific safety measures related to the activity.

         5. **Prepare Facilitator Guides**:
            - Create facilitator guides that outline the roles and responsibilities of the activity facilitators.
            - Include step-by-step instructions, tips for engaging participants, and any other relevant information.

         6. **Design Scoring Systems**:
            - For competitive activities, design scoring systems that are fair and transparent.
            - Ensure that the scoring criteria are clearly communicated to all participants.

         7. **Plan Rotation Schedules**:
            - Develop rotation schedules for activities that involve multiple groups or teams.
            - Ensure that the schedules are efficient and allow for smooth transitions between activities.

         8. **Integrate with Other Plans**:
            - Ensure that the activity logistics align with the catering plan, transportation plan, and other relevant aspects of the event.
            - Coordinate with the technology plan to ensure that any necessary AV equipment or WiFi access is available.

         9. **Review and Finalize**:
            - Review all the logistics plans to ensure they are comprehensive and practical.
            - Make any necessary adjustments to ensure that the activities run smoothly and safely.

         Here is an example structure to guide your writing:
         - **Activity Name**: [Name of the activity]
         - **Equipment List**: [List of required equipment]
         - **Setup Instructions**: [Detailed setup instructions]
         - **Safety Protocols**: [Safety guidelines and emergency procedures]
         - **Facilitator Guide**: [Roles and responsibilities, step-by-step instructions]
         - **Scoring System**: [Scoring criteria and guidelines]
         - **Rotation Schedule**: [Schedule for team rotations]

         Ensure that each section is clearly outlined and provides all necessary information for the successful execution of the activity.

         Finally, compile all the logistics plans into a detailed document that can be easily referenced during the event planning and execution phases.
         """.strip()
        ),
        requirements=[
            "The activity logistics must include equipment lists, setup instructions, safety protocols, facilitator guides, scoring systems, and rotation schedules"
        ],
        user_variables={},
    )
    assert activity_logistics.value is not None, (
        'ERROR: task "activity_logistics" execution failed'
    )

    # 12. Create a transportation plan with charter buses, parking management, shuttle service, and contingency plans. - - TRANSPORTATION_PLAN
    transportation_plan = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to create a comprehensive transportation plan for the corporate team-building event. This plan should include charter buses, parking management, shuttle service, and contingency plans. Follow these steps to accomplish your task:

         1. **Review Gathered Information**:
            Start by reviewing the information gathered about the company, event requirements, and constraints. This will help you understand the specific needs and logistics of the event:
            <information_gathering>
            {{INFORMATION_GATHERING}}
            </information_gathering>

         2. **Understand the Event Schedule**:
            Familiarize yourself with the detailed event schedule to ensure that the transportation plan aligns with the timing and logistics of the event:
            <event_schedule>
            {{EVENT_SCHEDULE}}
            </event_schedule>

         3. **Develop Charter Bus Plan**:
            Create a plan for charter buses that includes:
            - Number of buses required based on the number of attendees.
            - Pick-up and drop-off locations.
            - Schedule for bus arrivals and departures.
            - Cost estimates and budget considerations.

         4. **Design Parking Management Plan**:
            Develop a parking management plan that includes:
            - Availability of parking spaces at the venue.
            - Designated parking areas for attendees, staff, and vendors.
            - Parking fees and payment methods.
            - Accessibility considerations for attendees with disabilities.

         5. **Create Shuttle Service Plan**:
            Outline a shuttle service plan that includes:
            - Routes and schedules for shuttles.
            - Pick-up and drop-off points.
            - Frequency of shuttle services.
            - Cost estimates and budget considerations.

         6. **Develop Contingency Plans**:
            Prepare contingency plans for potential issues such as:
            - Weather-related delays or cancellations.
            - Mechanical failures of buses or shuttles.
            - Unexpected changes in attendee numbers.
            - Alternative transportation options in case of emergencies.

         7. **Integrate with Other Logistics**:
            Ensure that the transportation plan is integrated with other event logistics, such as registration, catering, and activities. This will help ensure a smooth and coordinated event experience.

         8. **Finalize the Transportation Plan**:
            Compile all the information into a detailed transportation plan that includes all the elements mentioned above. Ensure that the plan is clear, organized, and easy to follow.

         Your final answer should be a comprehensive transportation plan that addresses all the requirements and considerations mentioned above.
         """.strip()
        ),
        requirements=[
            "The transportation plan must include charter buses, parking management, shuttle service, and contingency plans"
        ],
        user_variables={
            "INFORMATION_GATHERING": information_gathering.value,
            "EVENT_SCHEDULE": event_schedule.value,
        },
    )
    assert transportation_plan.value is not None, (
        'ERROR: task "transportation_plan" execution failed'
    )

    # 13. Design a registration process with online portal, on-site check-in, name tags, welcome packets, and staff training. - - REGISTRATION_PROCESS
    registration_process = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to design a comprehensive registration process for the corporate team-building event. This process should include an online portal, on-site check-in, name tags, welcome packets, and staff training. Follow these steps to accomplish your task:

         1. **Online Portal**:
               - Design an online registration portal that allows employees to register for the event.
               - Ensure the portal collects necessary information such as name, department, dietary restrictions, and any special needs.
               - Include a confirmation email feature that sends registration details and event information to participants.

         2. **On-Site Check-In**:
               - Develop a streamlined on-site check-in process to ensure smooth and efficient entry for all attendees.
               - Plan for check-in stations with clear signage and trained staff to assist participants.
               - Consider using technology such as QR codes or barcode scanners to expedite the check-in process.

         3. **Name Tags**:
               - Design and prepare name tags for all attendees.
               - Include the participant's name, department, and any additional information that may be relevant (e.g., dietary restrictions, accessibility needs).
               - Ensure name tags are durable, easy to read, and can be worn comfortably throughout the event.

         4. **Welcome Packets**:
               - Create welcome packets that include essential event information such as the schedule, map of the venue, and any necessary forms or documents.
               - Include a welcome letter from the event organizers or company executives.
               - Add promotional items or swag to the packets to enhance the participant experience.

         5. **Staff Training**:
               - Develop a training program for event staff to ensure they are prepared to handle registration, check-in, and any issues that may arise.
               - Train staff on the use of the online portal, check-in technology, and how to assist participants with special needs.
               - Conduct mock drills to simulate the registration process and identify any potential issues or areas for improvement.

         6. **Integration with Other Plans**:
               - Ensure the registration process aligns with the overall event schedule, transportation plan, and safety plan.
               - Coordinate with the technology plan to integrate any necessary tech solutions for registration and check-in.
               - Consider the accessibility plan to ensure the registration process is inclusive and accommodates all participants.

         7. **Contingency Planning**:
               - Develop a contingency plan for potential issues such as technical difficulties with the online portal, high check-in volumes, or last-minute registrations.
               - Ensure staff are trained to handle these scenarios and have backup plans in place.

         8. **Review and Finalize**:
               - Review the registration process to ensure it meets all requirements and aligns with the event's goals.
               - Make any necessary adjustments and finalize the registration process.

         Use the information gathered in the previous steps to guide your design. Ensure the registration process is user-friendly, efficient, and aligns with the overall event strategy.

         <information_gathering>
         {{INFORMATION_GATHERING}}
         </information_gathering>

         <planning_timeline>
         {{PLANNING_TIMELINE}}
         </planning_timeline>

         <budget_breakdown>
         {{BUDGET_BREAKDOWN}}
         </budget_breakdown>

         <marketing_strategy>
         {{MARKETING_STRATEGY}}
         </marketing_strategy>

         <venue_selection>
         {{VENUE_SELECTION}}
         </venue_selection>

         <event_schedule>
         {{EVENT_SCHEDULE}}
         </event_schedule>

         <icebreaker_plans>
         {{ICEBREAKER_PLANS}}
         </icebreaker_plans>

         <team_formation>
         {{TEAM_FORMATION}}
         </team_formation>

         <workshop_content>
         {{WORKSHOP_CONTENT}}
         </workshop_content>

         <catering_plan>
         {{CATERING_PLAN}}
         </catering_plan>

         <activity_logistics>
         {{ACTIVITY_LOGISTICS}}
         </activity_logistics>

         <transportation_plan>
         {{TRANSPORTATION_PLAN}}
         </transportation_plan>
         """.strip()
        ),
        requirements=[
            "The registration process must include online portal, on-site check-in, name tags, welcome packets with schedule and swag, and staff training"
        ],
        user_variables={
            "INFORMATION_GATHERING": information_gathering.value,
            "PLANNING_TIMELINE": planning_timeline.value,
            "BUDGET_BREAKDOWN": budget_breakdown.value,
            "MARKETING_STRATEGY": marketing_strategy.value,
            "VENUE_SELECTION": venue_selection.value,
            "EVENT_SCHEDULE": event_schedule.value,
            "ICEBREAKER_PLANS": icebreaker_plans.value,
            "TEAM_FORMATION": team_formation.value,
            "WORKSHOP_CONTENT": workshop_content.value,
            "CATERING_PLAN": catering_plan.value,
            "ACTIVITY_LOGISTICS": activity_logistics.value,
            "TRANSPORTATION_PLAN": transportation_plan.value,
        },
    )
    assert registration_process.value is not None, (
        'ERROR: task "registration_process" execution failed'
    )

    # 14. Develop a comprehensive safety plan including insurance, first aid stations, evacuation procedures, weather contingencies, food safety, security personnel, and incident reporting. - - SAFETY_PLAN
    safety_plan = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to develop a comprehensive safety plan for the corporate team-building event. This plan should ensure the safety and well-being of all participants and staff. Follow these steps to accomplish your task:

         1. **Understand the Event Details**:
            Review the gathered information about the company, event requirements, and constraints from the previous step:
            <information_gathering>
            {{INFORMATION_GATHERING}}
            </information_gathering>

         2. **Identify Key Safety Components**:
            The safety plan should include the following components:
            - **Insurance**: Ensure that the event has adequate insurance coverage for all activities and participants.
            - **First Aid Stations**: Plan for the setup of first aid stations at strategic locations within the venue.
            - **Evacuation Procedures**: Develop clear evacuation procedures in case of emergencies.
            - **Weather Contingencies**: Prepare contingency plans for adverse weather conditions.
            - **Food Safety**: Implement food safety measures, including allergen labeling and proper handling procedures.
            - **Security Personnel**: Arrange for security personnel to be present throughout the event.
            - **Incident Reporting**: Establish a system for reporting and managing incidents during the event.

         3. **Create Detailed Plans for Each Component**:
            - **Insurance**: Specify the type and amount of insurance required, and ensure that all vendors and participants are covered.
            - **First Aid Stations**: Determine the number and location of first aid stations, and ensure that they are staffed by qualified medical personnel.
            - **Evacuation Procedures**: Develop a clear and concise evacuation plan, including designated assembly points and communication protocols.
            - **Weather Contingencies**: Prepare alternative plans for outdoor activities in case of bad weather, including indoor alternatives and rescheduling options.
            - **Food Safety**: Work with caterers to ensure that all food is prepared and served safely, with clear labeling of allergens.
            - **Security Personnel**: Hire and brief security personnel on their roles and responsibilities, including crowd control and emergency response.
            - **Incident Reporting**: Create a system for reporting incidents, including forms and protocols for documenting and addressing issues.

         4. **Review and Finalize the Safety Plan**:
            Ensure that the safety plan is comprehensive and addresses all potential risks and contingencies. Make any necessary adjustments to ensure that the plan is practical and effective.

         5. **Output the Safety Plan**:
            Provide the detailed safety plan as your final answer. The plan should be clear, concise, and easy to follow, ensuring that all safety measures are in place for a successful event.
         """.strip()
        ),
        requirements=[
            "The safety plan must include insurance, first aid stations, evacuation procedures, weather contingencies, food safety with allergen labeling, security personnel, and incident reporting"
        ],
        user_variables={"INFORMATION_GATHERING": information_gathering.value},
    )
    assert safety_plan.value is not None, 'ERROR: task "safety_plan" execution failed'

    # 15. Create a vendor management strategy with selection criteria, contracts, payment schedules, and evaluation forms. - - VENDOR_MANAGEMENT
    vendor_management = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to create a comprehensive vendor management strategy for the corporate team-building event. This strategy should include selection criteria, contracts, payment schedules, and evaluation forms. Follow these steps to accomplish your task:

         1. **Understand the Requirements**:
            Review the gathered information about the company, event requirements, and constraints from the previous step:
            <information_gathering>
            {{INFORMATION_GATHERING}}
            </information_gathering>

         2. **Define Selection Criteria**:
            Develop a set of selection criteria for vendors. Consider factors such as:
            - Experience and reputation
            - Cost-effectiveness
            - Availability and capacity
            - Quality of services or products
            - Compatibility with event goals and values
            - Sustainability practices
            - Customer reviews and references

         3. **Create Contract Templates**:
            Design standard contract templates that outline:
            - Scope of work
            - Deliverables
            - Timeline and milestones
            - Payment terms
            - Cancellation and refund policies
            - Liability and insurance requirements
            - Confidentiality and data protection clauses

         4. **Develop Payment Schedules**:
            Establish payment schedules for each vendor category (e.g., venue rental, catering, entertainment, transportation, technology). Consider:
            - Deposit amounts and due dates
            - Progress payments tied to milestones
            - Final payments upon completion
            - Payment methods and invoicing procedures

         5. **Design Evaluation Forms**:
            Create evaluation forms to assess vendor performance. Include sections for:
            - Quality of services or products
            - Timeliness and reliability
            - Professionalism and communication
            - Adherence to budget and contract terms
            - Customer feedback and satisfaction
            - Recommendations for future events

         6. **Compile the Vendor Management Strategy**:
            Combine all the above elements into a cohesive vendor management strategy document. Ensure it is clear, detailed, and aligned with the event's goals and constraints.

         Your final answer should be a comprehensive vendor management strategy document that includes selection criteria, contract templates, payment schedules, and evaluation forms.
         """.strip()
        ),
        requirements=[
            "The vendor management strategy must include selection criteria, contracts, payment schedules, and evaluation forms"
        ],
        user_variables={"INFORMATION_GATHERING": information_gathering.value},
    )
    assert vendor_management.value is not None, (
        'ERROR: task "vendor_management" execution failed'
    )

    # 16. Design a technology plan with event app, AV equipment, WiFi, photography services, live streaming, and tech support. - - TECHNOLOGY_PLAN
    technology_plan = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to design a comprehensive technology plan for the corporate team-building event. This plan should ensure seamless integration of technology to enhance the event experience, facilitate smooth operations, and support all planned activities.

         To accomplish this, follow these steps:

         1. **Event App**:
               - Design an event app that will serve as a central hub for scheduling, messaging, and real-time updates.
               - Include features such as personalized schedules, team assignments, interactive maps, and push notifications.
               - Ensure the app is user-friendly and accessible on both iOS and Android platforms.

         2. **AV Equipment**:
               - Plan for all necessary audio-visual equipment, including microphones, speakers, projectors, and screens.
               - Ensure that the AV setup is compatible with the venue's infrastructure and meets the requirements for presentations, workshops, and the opening ceremony.

         3. **WiFi**:
               - Arrange for robust WiFi coverage capable of supporting 300+ devices simultaneously.
               - Work with the venue and internet service providers to ensure reliable connectivity throughout the event.

         4. **Photography Services**:
               - Hire professional photographers to capture key moments during the event.
               - Plan for photo backdrops, designated photo areas, and a system for sharing photos with participants post-event.

         5. **Live Streaming**:
               - Set up live streaming capabilities for any sessions that need to be broadcast to remote participants or recorded for future reference.
               - Ensure high-quality video and audio streaming with minimal latency.

         6. **Tech Support**:
               - Assign a dedicated tech support team to troubleshoot any issues that arise during the event.
               - Provide clear contact information and a quick response protocol for technical difficulties.

         7. **Integration with Other Plans**:
               - Coordinate with the marketing and communication strategy to ensure the event app and other technology tools are promoted effectively.
               - Align with the registration process to integrate the event app with the online portal and on-site check-in.

         8. **Safety and Contingency**:
               - Include backup plans for technology failures, such as alternative AV setups and backup internet connections.
               - Ensure that all technology plans comply with the safety plan and contingency planning.

         9. **Budget Considerations**:
               - Estimate the costs for each technology component and ensure they fit within the overall budget.
               - Prioritize cost-effective solutions that meet the event's requirements without compromising quality.

         10. **Documentation**:
               - Create detailed documentation for each technology component, including setup instructions, user guides, and troubleshooting tips.
               - Share this documentation with relevant stakeholders, including event staff, vendors, and participants.

         Use the gathered information and previous steps to inform your technology plan. Ensure that all technology solutions are scalable, reliable, and enhance the overall event experience.

         Finally, compile your technology plan into a clear and concise document that can be integrated into the comprehensive event strategy document.
         """.strip()
        ),
        requirements=[
            "The technology plan must include event app for scheduling and messaging, AV equipment, WiFi for 300+ connections, photography services, live streaming, and tech support",
            "All planning must follow best practices, create memorable experiences, accommodate diverse needs, stay within budget, prioritize safety, incorporate sustainable practices, leverage technology, build excitement, and measure success",
        ],
        user_variables={},
    )
    assert technology_plan.value is not None, (
        'ERROR: task "technology_plan" execution failed'
    )

    # 17. Develop a decoration strategy with signage, team colors, photo backdrops, and sustainable options. - - DECORATION_STRATEGY
    decoration_strategy = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to develop a comprehensive decoration strategy for the corporate team-building event. This strategy should focus on creating an engaging and visually appealing environment that aligns with the event's goals and themes. Follow these steps to accomplish your task:

         1. **Understand the Event Requirements**:
            Review the gathered information about the company, event requirements, and constraints from the previous step:
            <information_gathering>
            {{INFORMATION_GATHERING}}
            </information_gathering>

         2. **Review the Event Schedule and Theme**:
            Ensure that the decoration strategy aligns with the event schedule and overall theme. Refer to the event schedule and other relevant plans:
            <event_schedule>
            {{EVENT_SCHEDULE}}
            </event_schedule>

         3. **Develop Signage Strategy**:
            Create a plan for signage that includes:
            - Welcome signs at the entrance
            - Directional signs for different event areas
            - Informational signs for activities and workshops
            - Team identification signs
            Ensure that all signage is clear, visually appealing, and consistent with the event's branding.

         4. **Design Team Colors and Identification**:
            Develop a strategy for using team colors to identify different teams. This can include:
            - Color-coded name tags and lanyards
            - Team-specific decorations and signage
            - Color-coordinated table settings and seating arrangements
            Ensure that the team colors are easily distinguishable and contribute to a cohesive visual theme.

         5. **Create Photo Backdrops**:
            Design photo backdrops that are visually appealing and align with the event's theme. Consider:
            - Themed backdrops for different event areas
            - Customizable backdrops that can be personalized with team names or slogans
            - Backdrops that incorporate the company's branding and colors
            Ensure that the photo backdrops are easily accessible and encourage participants to take photos.

         6. **Incorporate Sustainable Options**:
            Develop a strategy for incorporating sustainable decoration options. This can include:
            - Using reusable or recyclable materials
            - Choosing eco-friendly signage and backdrops
            - Implementing digital signage where possible
            - Ensuring proper waste management and recycling practices
            Ensure that the sustainable options are practical and do not compromise the visual appeal of the decorations.

         7. **Review and Finalize the Strategy**:
            Ensure that the decoration strategy is comprehensive and aligns with the event's goals and requirements. Make any necessary adjustments to ensure that the strategy is practical and effective.

         Finally, write the decoration strategy document that includes all the above elements. This document will serve as a guide for implementing the decoration strategy during the event.
         """.strip()
        ),
        requirements=[
            "The decoration strategy must include signage, team colors, photo backdrops, and sustainable options"
        ],
        user_variables={
            "INFORMATION_GATHERING": information_gathering.value,
            "EVENT_SCHEDULE": event_schedule.value,
        },
    )
    assert decoration_strategy.value is not None, (
        'ERROR: task "decoration_strategy" execution failed'
    )

    # 18. Create an entertainment plan with live band or DJ, emcee, interactive activities, and evening games. - - ENTERTAINMENT_PLAN
    entertainment_plan = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to create an entertainment plan for the corporate team-building event. The plan should include a live band or DJ, an emcee, interactive activities, and evening games. Follow these steps to accomplish your task:

         1. **Review the Event Schedule**:
            Ensure that the entertainment plan aligns with the event schedule. You can refer to the event schedule from the previous step:
            <event_schedule>
            {{EVENT_SCHEDULE}}
            </event_schedule>

         2. **Select Entertainment Options**:
            Choose between a live band or a DJ for the event. Consider the preferences of the attendees and the overall theme of the event. Additionally, decide on an emcee who will guide the event and engage the audience.

         3. **Plan Interactive Activities**:
            Develop a list of interactive activities that will engage the attendees. These activities should be fun, inclusive, and align with the team-building goals of the event. Examples include trivia games, karaoke, and dance-offs.

         4. **Design Evening Games**:
            Create a list of evening games that will keep the attendees entertained and engaged. These games should be suitable for the time of day and the overall atmosphere of the event. Examples include casino night, poker tournaments, and photo booths.

         5. **Coordinate with Other Plans**:
            Ensure that the entertainment plan coordinates with other aspects of the event, such as the catering plan, technology plan, and decoration strategy. You can refer to the following plans from the previous steps:
            <catering_plan>
            {{CATERING_PLAN}}
            </catering_plan>
            <technology_plan>
            {{TECHNOLOGY_PLAN}}
            </technology_plan>
            <decoration_strategy>
            {{DECORATION_STRATEGY}}
            </decoration_strategy>

         6. **Create a Detailed Entertainment Plan**:
            Write a detailed entertainment plan that includes the following information:
            - **Live Band or DJ**: Specify the type of entertainment, the duration, and any specific requirements.
            - **Emcee**: Provide details about the emcee, including their role and responsibilities.
            - **Interactive Activities**: List the interactive activities, their duration, and any necessary equipment or setup.
            - **Evening Games**: Describe the evening games, their duration, and any specific instructions or rules.

         Ensure that the entertainment plan is comprehensive and covers all aspects of the event. The plan should be clear, concise, and easy to follow.

         Finally, provide the detailed entertainment plan as your answer.
         """.strip()
        ),
        requirements=[
            "The entertainment plan must include live band or DJ, emcee, interactive activities, and evening games",
            "All planning must follow best practices, create memorable experiences, accommodate diverse needs, stay within budget, prioritize safety, incorporate sustainable practices, leverage technology, build excitement, and measure success",
        ],
        user_variables={
            "EVENT_SCHEDULE": event_schedule.value,
            "CATERING_PLAN": catering_plan.value,
            "TECHNOLOGY_PLAN": technology_plan.value,
            "DECORATION_STRATEGY": decoration_strategy.value,
        },
    )
    assert entertainment_plan.value is not None, (
        'ERROR: task "entertainment_plan" execution failed'
    )

    # 19. Design a prize strategy with team prizes, individual awards, participation gifts, and raffle prizes. - - PRIZE_STRATEGY
    prize_strategy = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to design a prize strategy for the corporate team-building event. The prize strategy should include team prizes, individual awards, participation gifts, and raffle prizes. Follow these steps to accomplish your task:

         1. **Understand the Event Context**:
            Review the gathered information about the company, event requirements, and constraints from the previous step:
            <information_gathering>
            {{INFORMATION_GATHERING}}
            </information_gathering>

         2. **Define Prize Categories**:
            Identify the different categories of prizes needed for the event. These should include:
            - **Team Prizes**: Prizes for the top-performing teams, such as first, second, and third place.
            - **Individual Awards**: Recognition for outstanding individual performances or contributions.
            - **Participation Gifts**: Small gifts or tokens of appreciation for all attendees.
            - **Raffle Prizes**: Prizes to be raffled off during the event, which can be won by any attendee.

         3. **Determine Prize Values**:
            Based on the budget breakdown and the overall event budget, determine the appropriate values for each prize category. Ensure that the prize strategy aligns with the total estimated budget:
            <budget_breakdown>
            {{BUDGET_BREAKDOWN}}
            </budget_breakdown>

         4. **Plan Team Prizes**:
            Design a tiered prize structure for team prizes. For example:
            - **First Place Team Prize**: $500 value
            - **Second Place Team Prize**: $300 value
            - **Third Place Team Prize**: $200 value

         5. **Plan Individual Awards**:
            Identify criteria for individual awards, such as "Most Innovative Idea," "Best Team Player," or "Outstanding Leadership." Determine the type and value of each individual award.

         6. **Plan Participation Gifts**:
            Choose small, meaningful gifts that can be given to all attendees. These could include branded merchandise, gift cards, or other tokens of appreciation.

         7. **Plan Raffle Prizes**:
            Select a variety of raffle prizes that appeal to a broad range of attendees. Consider including high-value items, experiences, or other exciting prizes.

         8. **Integrate with Event Schedule**:
            Ensure that the prize strategy aligns with the event schedule, including when and how prizes will be awarded. Review the event schedule from the previous step:
            <event_schedule>
            {{EVENT_SCHEDULE}}
            </event_schedule>

         9. **Document the Prize Strategy**:
            Compile all the information into a detailed prize strategy document. Include the following sections:
            - **Team Prizes**: Description and value of each team prize.
            - **Individual Awards**: Criteria and description of each individual award.
            - **Participation Gifts**: Description and distribution plan for participation gifts.
            - **Raffle Prizes**: Description and selection process for raffle prizes.

         10. **Review and Finalize**:
               Review the prize strategy to ensure it meets all event requirements and aligns with the overall event goals. Make any necessary adjustments to ensure clarity and completeness.

         Your final answer should be a detailed prize strategy document that includes all the above information.
         """.strip()
        ),
        requirements=[
            "The prize strategy must include team prizes at $500, $300, and $200, individual awards, participation gifts, and raffle prizes"
        ],
        user_variables={
            "INFORMATION_GATHERING": information_gathering.value,
            "BUDGET_BREAKDOWN": budget_breakdown.value,
            "EVENT_SCHEDULE": event_schedule.value,
        },
    )
    assert prize_strategy.value is not None, (
        'ERROR: task "prize_strategy" execution failed'
    )

    # 20. Develop a post-event evaluation plan with participant survey, debriefing sessions, budget reconciliation, photo compilation, impact assessment, and thank you communications. - - POST_EVENT_EVALUATION
    post_event_evaluation = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to develop a comprehensive post-event evaluation plan for the corporate team-building event. This plan should include the following components:

         1. **Participant Survey**:
            - Design a survey to be sent within 24 hours of the event.
            - Include questions that assess the overall experience, specific activities, workshops, catering, and any areas for improvement.
            - Ensure the survey is concise, easy to complete, and covers all key aspects of the event.

         2. **Debriefing Sessions**:
            - Plan debriefing sessions with the event planning team and key stakeholders.
            - Schedule these sessions within a week after the event to discuss what went well, what could be improved, and any lessons learned.
            - Document the outcomes of these sessions for future reference.

         3. **Budget Reconciliation**:
            - Compare the actual expenses with the estimated budget.
            - Identify any budget overruns or savings.
            - Provide a detailed report on the financial performance of the event.

         4. **Photo Compilation**:
            - Gather all photos taken during the event.
            - Create a digital album or slideshow to share with participants and stakeholders.
            - Ensure the photos are organized and easily accessible.

         5. **Impact Assessment**:
            - Evaluate the impact of the event on team cohesion, cross-departmental collaboration, and morale.
            - Use feedback from the participant survey and debriefing sessions to assess the event's success.
            - Provide a summary of the event's overall impact and any measurable outcomes.

         6. **Thank You Communications**:
            - Prepare thank-you messages for all participants, vendors, and stakeholders.
            - Include a brief summary of the event's success and express gratitude for their contributions.
            - Send these communications within a week after the event.

         To accomplish this task, follow these steps:

         1. **Review Event Details**:
            - Refer to the gathered information about the company, event requirements, and constraints:
            <information_gathering>
            {{INFORMATION_GATHERING}}
            </information_gathering>

         2. **Design the Participant Survey**:
            - Create a survey that covers all key aspects of the event.
            - Ensure the survey is user-friendly and can be completed quickly.

         3. **Plan Debriefing Sessions**:
            - Schedule sessions with the event planning team and key stakeholders.
            - Prepare an agenda for each session to ensure productive discussions.

         4. **Conduct Budget Reconciliation**:
            - Compare actual expenses with the estimated budget.
            - Identify any discrepancies and document the findings.

         5. **Compile Photos**:
            - Gather all photos from the event.
            - Organize them into a digital album or slideshow.

         6. **Assess Impact**:
            - Use feedback from the survey and debriefing sessions to evaluate the event's impact.
            - Provide a summary of the event's success and any measurable outcomes.

         7. **Prepare Thank You Communications**:
            - Draft thank-you messages for participants, vendors, and stakeholders.
            - Include a brief summary of the event's success and express gratitude.

         8. **Compile the Post-Event Evaluation Plan**:
            - Combine all the components into a comprehensive post-event evaluation plan.
            - Ensure the plan is clear, detailed, and ready for implementation.

         Your final answer should be the complete post-event evaluation plan, including all the components mentioned above.
         """.strip()
        ),
        requirements=[
            "The post-event evaluation must include participant survey within 24 hours, debriefing sessions, budget reconciliation, photo compilation, impact assessment, and thank you communications"
        ],
        user_variables={"INFORMATION_GATHERING": information_gathering.value},
    )
    assert post_event_evaluation.value is not None, (
        'ERROR: task "post_event_evaluation" execution failed'
    )

    # 21. Create an accessibility plan with ADA compliance, interpreters, activity modifications, dietary accommodation, and inclusive team formation. - - ACCESSIBILITY_PLAN
    accessibility_plan = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to create an accessibility plan for the corporate team-building event. This plan should ensure that the event is inclusive and accommodates the diverse needs of all attendees. Follow these steps to accomplish your task:

         1. **Understand the Requirements**:
            Review the gathered information about the company and event requirements to understand the specific needs and constraints related to accessibility:
            <information_gathering>
            {{INFORMATION_GATHERING}}
            </information_gathering>

         2. **ADA Compliance**:
            Ensure that the event venue and all activities comply with the Americans with Disabilities Act (ADA) standards. This includes:
            - Accessible entry points and pathways
            - Adequate space for mobility devices
            - Accessible restrooms
            - Clear signage and directions

         3. **Interpreters and Assistive Technology**:
            Plan for the provision of interpreters for attendees who are deaf or hard of hearing. Additionally, consider any assistive technology that may be required, such as:
            - Hearing assistance devices
            - Visual aids
            - Braille materials

         4. **Activity Modifications**:
            Modify activities to ensure they are accessible to all attendees. This may include:
            - Providing alternative activities for those who cannot participate in certain events
            - Adjusting the difficulty level of activities
            - Ensuring that all activities can be performed by individuals with varying abilities

         5. **Dietary Accommodation**:
            Accommodate dietary restrictions and preferences by:
            - Offering a variety of food options that cater to different dietary needs (e.g., vegetarian, vegan, gluten-free, allergies)
            - Clearly labeling all food items with allergen information
            - Providing a pre-event survey to gather dietary information from attendees

         6. **Inclusive Team Formation**:
            Ensure that teams are formed inclusively by:
            - Mixing departments and seniority levels
            - Considering the abilities and preferences of each attendee
            - Providing clear team assignments and communication channels

         7. **Safety and Emergency Procedures**:
            Develop safety and emergency procedures that are accessible to all attendees. This includes:
            - Clear evacuation routes and procedures
            - Accessible first aid stations
            - Emergency contact information

         8. **Communication Plan**:
            Create a communication plan that ensures all attendees are informed about accessibility options and how to access them. This may include:
            - Including accessibility information in event materials
            - Providing a dedicated point of contact for accessibility-related questions
            - Using multiple communication channels (e.g., email, event app, Slack)

         9. **Review and Finalize**:
            Review the accessibility plan to ensure it meets all requirements and addresses the diverse needs of the attendees. Make any necessary adjustments and finalize the plan.

         Your final answer should be a comprehensive accessibility plan that includes all the elements outlined above.
         """.strip()
        ),
        requirements=[
            "The accessibility plan must include ADA compliance, interpreters, activity modifications, dietary accommodation, and inclusive team formation"
        ],
        user_variables={"INFORMATION_GATHERING": information_gathering.value},
    )
    assert accessibility_plan.value is not None, (
        'ERROR: task "accessibility_plan" execution failed'
    )

    # 22. Create a contingency plan for weather emergencies, vendor cancellations, technology failures, medical emergencies, and budget overruns. - - CONTINGENCY_PLANNING
    contingency_planning = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to create a comprehensive contingency plan for the corporate team-building event. This plan should address potential risks and ensure the event can proceed smoothly despite unforeseen circumstances. Follow these steps to accomplish your task:

         1. **Identify Potential Risks**:
            - **Weather Emergencies**: Consider the possibility of extreme weather conditions that could disrupt the event. Think about both indoor and outdoor activities.
            - **Vendor Cancellations**: Identify key vendors and the impact of their cancellation on the event.
            - **Technology Failures**: Consider the technology requirements for the event and potential failures that could occur.
            - **Medical Emergencies**: Plan for any medical emergencies that might arise during the event.
            - **Budget Overruns**: Identify areas where budget overruns might occur and plan for alternative solutions.

         2. **Develop Contingency Strategies**:
            - **Weather Emergencies**:
               - **Indoor Activities**: Have a backup plan for moving outdoor activities indoors.
               - **Outdoor Activities**: Consider alternative dates or locations if necessary.
               - **Communication**: Develop a communication plan to inform attendees of any changes due to weather.
            - **Vendor Cancellations**:
               - **Backup Vendors**: Identify and contract backup vendors for critical services.
               - **Alternative Solutions**: Develop alternative solutions for essential services.
            - **Technology Failures**:
               - **Backup Equipment**: Ensure backup AV equipment and tech support are available.
               - **Manual Processes**: Have manual processes in place in case of technology failures.
            - **Medical Emergencies**:
               - **First Aid Stations**: Ensure adequate first aid stations and trained personnel are available.
               - **Emergency Protocols**: Develop clear protocols for handling medical emergencies.
            - **Budget Overruns**:
               - **Cost-Saving Measures**: Identify areas where costs can be reduced without compromising the event.
               - **Contingency Fund**: Allocate a contingency fund to cover unexpected expenses.

         3. **Implement and Monitor**:
            - **Implementation**: Ensure all contingency plans are implemented as part of the overall event strategy.
            - **Monitoring**: Continuously monitor the event for any signs of the identified risks and be prepared to activate the contingency plans as needed.

         4. **Documentation**:
            - **Contingency Plan Document**: Create a detailed contingency plan document that includes all the strategies developed. This document should be part of the comprehensive event strategy document.

         5. **Review and Update**:
            - **Regular Reviews**: Regularly review and update the contingency plan as the event approaches to ensure it remains relevant and effective.

         Use the information gathered in the previous steps to inform your contingency planning. Ensure that all aspects of the event are covered and that the contingency plans are practical and feasible.

         Finally, compile the contingency plan into a clear and concise document that can be easily referenced during the event planning and execution phases.
         """.strip()
        ),
        requirements=[
            "The contingency planning must include weather emergencies, vendor cancellations, technology failures, medical emergencies, and budget overruns",
            "All planning must follow best practices, create memorable experiences, accommodate diverse needs, stay within budget, prioritize safety, incorporate sustainable practices, leverage technology, build excitement, and measure success",
        ],
        user_variables={},
    )
    assert contingency_planning.value is not None, (
        'ERROR: task "contingency_planning" execution failed'
    )

    # 23. Develop a sustainability plan with eco-friendly venue, local catering, reusable serving ware, digital communications, recyclable decorations, and waste management. - - SUSTAINABILITY_PLAN
    sustainability_plan = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to develop a comprehensive sustainability plan for the corporate team-building event. This plan should focus on minimizing the environmental impact of the event while ensuring a memorable and enjoyable experience for all participants.

         To accomplish this, follow these steps:

         1. **Understand the Event Requirements**:
            Review the gathered information about the company, event requirements, and constraints:
            <information_gathering>
            {{INFORMATION_GATHERING}}
            </information_gathering>

         2. **Identify Key Sustainability Areas**:
            Focus on the following key areas to create a sustainable event:
            - **Eco-friendly Venue**: Select a venue that has eco-friendly practices, such as energy-efficient lighting, waste reduction programs, and sustainable building materials.
            - **Local Catering**: Partner with local caterers who source ingredients locally and use sustainable practices. Ensure they offer diverse menu options that accommodate dietary restrictions.
            - **Reusable Serving Ware**: Use reusable or compostable serving ware to minimize waste. Avoid single-use plastics and encourage the use of reusable cups, plates, and cutlery.
            - **Digital Communications**: Utilize digital communications for invitations, reminders, and event information to reduce paper waste. Create a mobile event app for scheduling and messaging.
            - **Recyclable Decorations**: Choose decorations that are recyclable or made from sustainable materials. Avoid non-biodegradable materials and opt for eco-friendly alternatives.
            - **Waste Management**: Implement a comprehensive waste management plan that includes recycling stations, composting, and proper disposal of waste. Educate participants on the importance of waste reduction and proper disposal methods.

         3. **Develop Detailed Strategies**:
            For each key area, develop detailed strategies that outline the steps to be taken, the resources required, and the expected outcomes. Ensure that the strategies align with the overall event goals and constraints.

         4. **Create a Sustainability Timeline**:
            Integrate the sustainability plan into the overall event timeline. Ensure that sustainability initiatives are scheduled appropriately and that all necessary actions are taken in a timely manner.

         5. **Budget Considerations**:
            Review the budget breakdown to ensure that the sustainability plan is cost-effective and aligns with the total estimated budget:
            <budget_breakdown>
            {{BUDGET_BREAKDOWN}}
            </budget_breakdown>

         6. **Communication Plan**:
            Develop a communication plan to inform participants about the sustainability initiatives. Use the marketing and communication strategy to promote the event's eco-friendly practices:
            <marketing_strategy>
            {{MARKETING_STRATEGY}}
            </marketing_strategy>

         7. **Vendor Management**:
            Ensure that all vendors adhere to the sustainability plan. Include sustainability criteria in the vendor selection process and monitor their compliance throughout the event:
            <vendor_management>
            {{VENDOR_MANAGEMENT}}
            </vendor_management>

         8. **Post-Event Evaluation**:
            Include sustainability metrics in the post-event evaluation plan. Assess the effectiveness of the sustainability initiatives and identify areas for improvement in future events:
            <post_event_evaluation>
            {{POST_EVENT_EVALUATION}}
            </post_event_evaluation>

         9. **Contingency Planning**:
            Develop contingency plans for potential sustainability challenges, such as waste management issues or vendor non-compliance. Ensure that the contingency planning aligns with the overall event strategy:
            <contingency_planning>
            {{CONTINGENCY_PLANNING}}
            </contingency_planning>

         10. **Finalize the Sustainability Plan**:
               Compile all the information into a detailed sustainability plan that outlines the strategies, timelines, budget considerations, communication plans, vendor management, post-event evaluation, and contingency planning.

         Ensure that the sustainability plan is integrated into the comprehensive event strategy document and that all planning follows best practices, creates memorable experiences, accommodates diverse needs, stays within budget, prioritizes safety, incorporates sustainable practices, leverages technology, builds excitement, and measures success.
         """.strip()
        ),
        requirements=[
            "The sustainability plan must include eco-friendly venue, local catering, reusable serving ware, digital communications, recyclable decorations, and waste management",
            "All planning must follow best practices, create memorable experiences, accommodate diverse needs, stay within budget, prioritize safety, incorporate sustainable practices, leverage technology, build excitement, and measure success",
        ],
        user_variables={
            "INFORMATION_GATHERING": information_gathering.value,
            "BUDGET_BREAKDOWN": budget_breakdown.value,
            "MARKETING_STRATEGY": marketing_strategy.value,
            "VENDOR_MANAGEMENT": vendor_management.value,
            "POST_EVENT_EVALUATION": post_event_evaluation.value,
            "CONTINGENCY_PLANNING": contingency_planning.value,
        },
    )
    assert sustainability_plan.value is not None, (
        'ERROR: task "sustainability_plan" execution failed'
    )

    # 24. Compile all the information into a comprehensive event strategy document. - - EVENT_STRATEGY_DOCUMENT
    event_strategy_document = m.instruct(
        textwrap.dedent(
            R"""
         Your task is to compile all the information gathered and planned in the previous steps into a comprehensive event strategy document. This document will serve as the blueprint for executing the corporate team-building event.

         To accomplish this, follow these steps:

         1. **Review All Previous Steps**:
            Carefully review the information and plans developed in the previous steps. This includes:
            - Information Gathering: {{INFORMATION_GATHERING}}
            - Planning Timeline: {{PLANNING_TIMELINE}}
            - Budget Breakdown: {{BUDGET_BREAKDOWN}}
            - Marketing Strategy: {{MARKETING_STRATEGY}}
            - Venue Selection: {{VENUE_SELECTION}}
            - Event Schedule: {{EVENT_SCHEDULE}}
            - Icebreaker Plans: {{ICEBREAKER_PLANS}}
            - Team Formation: {{TEAM_FORMATION}}
            - Workshop Content: {{WORKSHOP_CONTENT}}
            - Catering Plan: {{CATERING_PLAN}}
            - Activity Logistics: {{ACTIVITY_LOGISTICS}}
            - Transportation Plan: {{TRANSPORTATION_PLAN}}
            - Registration Process: {{REGISTRATION_PROCESS}}
            - Safety Plan: {{SAFETY_PLAN}}
            - Vendor Management: {{VENDOR_MANAGEMENT}}
            - Technology Plan: {{TECHNOLOGY_PLAN}}
            - Decoration Strategy: {{DECORATION_STRATEGY}}
            - Entertainment Plan: {{ENTERTAINMENT_PLAN}}
            - Prize Strategy: {{PRIZE_STRATEGY}}
            - Post-Event Evaluation: {{POST_EVENT_EVALUATION}}
            - Accessibility Plan: {{ACCESSIBILITY_PLAN}}
            - Sustainability Plan: {{SUSTAINABILITY_PLAN}}
            - Contingency Planning: {{CONTINGENCY_PLANNING}}

         2. **Organize the Information**:
            Structure the event strategy document in a logical and coherent manner. Use clear headings and subheadings to organize the content. Here is a suggested structure:
            - **Executive Summary**: A brief overview of the event's objectives, key highlights, and expected outcomes.
            - **Event Overview**: Detailed description of the event, including its purpose, goals, and target audience.
            - **Planning Timeline**: The detailed planning timeline starting 6 months before the event with weekly milestones.
            - **Budget Breakdown**: Comprehensive breakdown of the budget for venue rental, catering, entertainment, transportation, technology, and total estimated budget.
            - **Marketing and Communication Strategy**: Detailed strategy for save-the-date announcements, formal invitations, reminder emails, internal campaigns, and other communication methods.
            - **Venue Selection**: Process for evaluating and selecting the best venue based on capacity, accessibility, parking, indoor/outdoor space, AV capabilities, WiFi, catering facilities, accommodations, and cost-effectiveness.
            - **Event Schedule**: Detailed event schedule from 8:00 AM to 8:00 PM including registration, breakfast, opening ceremony, icebreaker activities, workshops, lunch, team challenges, awards ceremony, and dinner with entertainment.
            - **Icebreaker Activities**: List of 15 icebreaker activities with detailed plans.
            - **Team Formation Strategy**: Strategy for forming diverse teams with a mix of departments and seniority, pre-event survey for dietary restrictions, and team assignments 2 weeks before the event.
            - **Workshop Content**: Content for leadership, innovation, communication, and wellness workshops.
            - **Catering Plan**: Detailed catering plan with breakfast, lunch, afternoon snacks, and dinner options.
            - **Activity Logistics**: Logistics for all planned activities including equipment lists, setup instructions, safety protocols, facilitator guides, scoring systems, and rotation schedules.
            - **Transportation Plan**: Plan for charter buses, parking management, shuttle service, and contingency plans.
            - **Registration Process**: Process for online portal, on-site check-in, name tags, welcome packets, and staff training.
            - **Safety Plan**: Comprehensive safety plan including insurance, first aid stations, evacuation procedures, weather contingencies, food safety, security personnel, and incident reporting.
            - **Vendor Management**: Strategy for vendor selection, contracts, payment schedules, and evaluation forms.
            - **Technology Plan**: Plan for event app, AV equipment, WiFi, photography services, live streaming, and tech support.
            - **Decoration Strategy**: Strategy for signage, team colors, photo backdrops, and sustainable options.
            - **Entertainment Plan**: Plan for live band or DJ, emcee, interactive activities, and evening games.
            - **Prize Strategy**: Strategy for team prizes, individual awards, participation gifts, and raffle prizes.
            - **Post-Event Evaluation**: Plan for participant survey, debriefing sessions, budget reconciliation, photo compilation, impact assessment, and thank you communications.
            - **Accessibility Plan**: Plan for ADA compliance, interpreters, activity modifications, dietary accommodation, and inclusive team formation.
            - **Sustainability Plan**: Plan for eco-friendly venue, local catering, reusable serving ware, digital communications, recyclable decorations, and waste management.
            - **Contingency Planning**: Plan for weather emergencies, vendor cancellations, technology failures, medical emergencies, and budget overruns.

         3. **Compile the Document**:
            Combine all the organized information into a single, cohesive document. Ensure that the document is well-structured, easy to read, and includes all necessary details for executing the event.

         4. **Review and Finalize**:
            Review the compiled document to ensure accuracy, completeness, and clarity. Make any necessary adjustments to ensure that the document meets all requirements and standards.

         5. **Output the Event Strategy Document**:
            Provide the final event strategy document as your answer. Ensure that the document is well-formatted and ready for use.
         """.strip()
        ),
        requirements=[
            "The event strategy document must include detailed planning timelines starting 6 months before the event with weekly milestones",
            "The budget breakdowns must include venue rental at $15,000-25,000, catering at $75-125 per person, entertainment and activities at $10,000-20,000, transportation at $3,000-8,000, technology at $5,000-10,000, and total estimated budget of $50,000-125,000",
            "The marketing and communication strategy must include save-the-date announcements 4 months in advance, formal invitations 2 months before, weekly reminder emails, internal campaigns, dedicated Slack channel, executive videos, and mobile event app",
            "The venue selection process must evaluate 10 venues based on capacity for 100-300 people, accessibility, parking, indoor/outdoor space, AV capabilities, WiFi for 300+ devices, catering facilities, accommodations, and cost-effectiveness",
            "The event schedule must include registration and breakfast at 8:00-9:00 AM, opening ceremony at 9:00-9:30 AM with CEO speech, icebreaker activities at 9:30-10:30 AM including human bingo and speed networking, morning workshops at 10:30 AM-12:30 PM with tracks on leadership, innovation, communication, and wellness, lunch at 12:30-2:00 PM with diverse menu options, afternoon team challenges at 2:00-5:00 PM including scavenger hunt, escape rooms, outdoor adventures, cooking competition, and sports tournaments, awards ceremony at 5:00-5:30 PM, and dinner with entertainment at 5:30-8:00 PM",
            "The icebreaker plans must include 15 options including human knot, marshmallow tower, obstacle course, pictionary, charades, and trust falls",
            "The team formation strategy must include diverse mix of departments and seniority in teams of 8-12 people, pre-event survey for dietary restrictions and preferences, and team assignments 2 weeks before with names and colors",
            "The workshop content must include leadership case studies, innovation brainstorming, communication exercises, and wellness meditation",
            "The catering plan must include breakfast coffee station and pastries, lunch buffet with salad bar and hot entrees, afternoon snacks, and dinner with three entree choices and full bar",
            "The activity logistics must include equipment lists, setup instructions, safety protocols, facilitator guides, scoring systems, and rotation schedules",
            "The transportation plan must include charter buses, parking management, shuttle service, and contingency plans",
            "The registration process must include online portal, on-site check-in, name tags, welcome packets with schedule and swag, and staff training",
            "The safety plan must include insurance, first aid stations, evacuation procedures, weather contingencies, food safety with allergen labeling, security personnel, and incident reporting",
            "The vendor management strategy must include selection criteria, contracts, payment schedules, and evaluation forms",
            "The technology plan must include event app for scheduling and messaging, AV equipment, WiFi for 300+ connections, photography services, live streaming, and tech support",
            "The decoration strategy must include signage, team colors, photo backdrops, and sustainable options",
            "The entertainment plan must include live band or DJ, emcee, interactive activities, and evening games",
            "The prize strategy must include team prizes at $500, $300, and $200, individual awards, participation gifts, and raffle prizes",
            "The post-event evaluation must include participant survey within 24 hours, debriefing sessions, budget reconciliation, photo compilation, impact assessment, and thank you communications",
            "The accessibility plan must include ADA compliance, interpreters, activity modifications, dietary accommodation, and inclusive team formation",
            "The sustainability plan must include eco-friendly venue, local catering, reusable serving ware, digital communications, recyclable decorations, and waste management",
            "The contingency planning must include weather emergencies, vendor cancellations, technology failures, medical emergencies, and budget overruns",
            "All planning must follow best practices, create memorable experiences, accommodate diverse needs, stay within budget, prioritize safety, incorporate sustainable practices, leverage technology, build excitement, and measure success",
        ],
        user_variables={
            "INFORMATION_GATHERING": information_gathering.value,
            "PLANNING_TIMELINE": planning_timeline.value,
            "BUDGET_BREAKDOWN": budget_breakdown.value,
            "MARKETING_STRATEGY": marketing_strategy.value,
            "VENUE_SELECTION": venue_selection.value,
            "EVENT_SCHEDULE": event_schedule.value,
            "ICEBREAKER_PLANS": icebreaker_plans.value,
            "TEAM_FORMATION": team_formation.value,
            "WORKSHOP_CONTENT": workshop_content.value,
            "CATERING_PLAN": catering_plan.value,
            "ACTIVITY_LOGISTICS": activity_logistics.value,
            "TRANSPORTATION_PLAN": transportation_plan.value,
            "REGISTRATION_PROCESS": registration_process.value,
            "SAFETY_PLAN": safety_plan.value,
            "VENDOR_MANAGEMENT": vendor_management.value,
            "TECHNOLOGY_PLAN": technology_plan.value,
            "DECORATION_STRATEGY": decoration_strategy.value,
            "ENTERTAINMENT_PLAN": entertainment_plan.value,
            "PRIZE_STRATEGY": prize_strategy.value,
            "POST_EVENT_EVALUATION": post_event_evaluation.value,
            "ACCESSIBILITY_PLAN": accessibility_plan.value,
            "SUSTAINABILITY_PLAN": sustainability_plan.value,
            "CONTINGENCY_PLANNING": contingency_planning.value,
        },
    )
    assert event_strategy_document.value is not None, (
        'ERROR: task "event_strategy_document" execution failed'
    )

    final_answer = event_strategy_document.value

    print(final_answer)
