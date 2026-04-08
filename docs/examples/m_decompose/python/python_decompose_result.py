# pytest: skip_always
import textwrap

import mellea

# Note: This is an example of an intermediary result from using decompose in python_decompose_example.py, not an example of how to use decompose.


m = mellea.start_session()


# 1. Create a catchy title for the blog post about the benefits of morning exercise. - - BLOG_TITLE
blog_title = m.instruct(
    textwrap.dedent(
        R"""
        Your task is to create a catchy title for a blog post about the benefits of morning exercise. Follow these steps to accomplish your task:

        1. **Understand the Topic**: The blog post will focus on the benefits of morning exercise. The title should be engaging and clearly convey the main topic of the post.

        2. **Identify Key Elements**: Consider the key elements that make morning exercise beneficial. These could include improved mood, increased energy, better focus, and enhanced metabolism.

        3. **Use Power Words**: Incorporate power words that evoke curiosity, excitement, or a sense of urgency. Examples include "Boost," "Transform," "Unlock," "Energize," and "Revitalize."

        4. **Keep It Concise**: The title should be short and to the point, ideally between 5 to 10 words. It should be easy to read and remember.

        5. **Make It Action-Oriented**: Use verbs that encourage action, such as "Start," "Jumpstart," "Kickstart," or "Ignite."

        6. **Consider SEO**: Think about common search terms related to morning exercise. Including relevant keywords can help improve the post's visibility.

        7. **Examples for Inspiration**:
           - "Jumpstart Your Day: The Power of Morning Exercise"
           - "Energize Your Mornings: Unlock the Benefits of Morning Exercise"
           - "Transform Your Day with Morning Exercise"
           - "Boost Your Energy: The Magic of Morning Workouts"
           - "Revitalize Your Mornings: The Benefits of Morning Exercise"

        8. **Create the Title**: Based on the above guidelines, create a catchy and engaging title for the blog post. Ensure it captures the essence of the topic and entices readers to click and read more.

        Your final answer should be only the title text.
        """.strip()
    ),
    requirements=["Include a catchy title"],
)
assert blog_title.value is not None, 'ERROR: task "blog_title" execution failed'

# 2. Write an introduction paragraph that sets the stage for the blog post. - - INTRODUCTION
introduction = m.instruct(
    textwrap.dedent(
        R"""
        Your task is to write an engaging introduction paragraph for a blog post about the benefits of morning exercise. The introduction should set the stage for the blog post, capturing the reader's attention and providing a brief overview of what will be discussed.

        To accomplish this, follow these steps:

        1. **Understand the Context**:
           - The blog post is about the benefits of morning exercise.
           - The title of the blog post is: {{BLOG_TITLE}}

        2. **Craft the Introduction**:
           - Start with a hook that grabs the reader's attention. This could be a question, a surprising fact, or a relatable scenario.
           - Briefly introduce the topic of morning exercise and why it is important.
           - Provide a smooth transition to the main benefits that will be discussed in the blog post.

        3. **Ensure Engagement**:
           - Use a conversational and engaging tone to connect with the readers.
           - Keep the introduction concise and to the point, ideally between 3 to 5 sentences.

        Here is an example structure to guide your writing:
        - **Sentence 1**: Hook to grab the reader's attention.
        - **Sentence 2**: Introduce the topic of morning exercise.
        - **Sentence 3**: Briefly mention the benefits that will be discussed.
        - **Sentence 4**: Transition to the main content of the blog post.

        Ensure that the introduction flows naturally and sets the stage for the rest of the blog post. You should write only the introduction paragraph, do not include the guidance structure.
        """.strip()
    ),
    requirements=["Include an introduction paragraph"],
    user_variables={"BLOG_TITLE": blog_title.value},
)
assert introduction.value is not None, 'ERROR: task "introduction" execution failed'

# 3. Identify and explain three main benefits of morning exercise with detailed explanations. - - BENEFITS
benefits = m.instruct(
    textwrap.dedent(
        R"""
        Your task is to identify and explain three main benefits of morning exercise with detailed explanations. Follow these steps to accomplish your task:

        First, review the title and introduction created in the previous steps to understand the context and tone of the blog post:
        <title>
        {{BLOG_TITLE}}
        </title>
        <introduction>
        {{INTRODUCTION}}
        </introduction>

        Next, research and identify three main benefits of morning exercise. These benefits should be supported by evidence or expert opinions to ensure credibility.

        For each benefit, provide a detailed explanation that includes:
        - The specific benefit of morning exercise
        - How this benefit positively impacts health, well-being, or daily life
        - Any relevant studies, expert opinions, or personal anecdotes that support the benefit

        Ensure that the explanations are clear, concise, and engaging to keep the reader interested.

        Finally, present the three main benefits with their detailed explanations in a structured format that can be easily integrated into the blog post.
        """.strip()
    ),
    requirements=["Include three main benefits with explanations"],
    user_variables={"BLOG_TITLE": blog_title.value, "INTRODUCTION": introduction.value},
)
assert benefits.value is not None, 'ERROR: task "benefits" execution failed'

# 4. Write a conclusion that encourages readers to start their morning exercise routine. - - CONCLUSION
conclusion = m.instruct(
    textwrap.dedent(
        R"""
        Your task is to write a compelling conclusion for a blog post about the benefits of morning exercise. The conclusion should encourage readers to start their morning exercise routine. Follow these steps to accomplish your task:

        First, review the title and introduction of the blog post to understand the context and tone:
        <title>
        {{BLOG_TITLE}}
        </title>
        <introduction>
        {{INTRODUCTION}}
        </introduction>

        Next, consider the three main benefits of morning exercise that have been previously identified and explained:
        <benefits>
        {{BENEFITS}}
        </benefits>

        Use the information from the title, introduction, and benefits to craft a conclusion that:
        1. Summarizes the key points discussed in the blog post.
        2. Reinforces the importance of morning exercise.
        3. Encourages readers to take action and start their morning exercise routine.
        4. Maintains a positive and motivating tone.

        Ensure the conclusion is concise, engaging, and leaves readers feeling inspired to make a change in their daily routine.

        Finally, write the conclusion paragraph that encourages readers to start their morning exercise routine.
        """.strip()
    ),
    requirements=[
        "Include a conclusion that encourages readers to start their morning exercise routine"
    ],
    user_variables={
        "BLOG_TITLE": blog_title.value,
        "INTRODUCTION": introduction.value,
        "BENEFITS": benefits.value,
    },
)
assert conclusion.value is not None, 'ERROR: task "conclusion" execution failed'

# 5. Compile the title, introduction, three main benefits, and conclusion into a single cohesive blog post. - - FINAL_BLOG_POST
final_blog_post = m.instruct(
    textwrap.dedent(
        R"""
        Your task is to compile the title, introduction, three main benefits, and conclusion into a single cohesive blog post about the benefits of morning exercise.

        To accomplish this, follow these steps:

        1. **Review the Components**:
           Carefully review the title, introduction, three main benefits, and conclusion that have been generated in the previous steps. These components are provided below:

           <blog_title>
           {{BLOG_TITLE}}
           </blog_title>

           <introduction>
           {{INTRODUCTION}}
           </introduction>

           <benefits>
           {{BENEFITS}}
           </benefits>

           <conclusion>
           {{CONCLUSION}}
           </conclusion>

        2. **Structure the Blog Post**:
           Organize the components into a well-structured blog post. The structure should include:
           - The catchy title at the beginning.
           - The introduction paragraph that sets the stage for the blog post.
           - The three main benefits with detailed explanations.
           - The conclusion that encourages readers to start their morning exercise routine.

        3. **Ensure Cohesion**:
           Make sure the blog post flows smoothly from one section to the next. The transitions between the introduction, benefits, and conclusion should be natural and logical.

        4. **Check for Consistency**:
           Verify that the tone and style are consistent throughout the blog post. Ensure that the language used in the title, introduction, benefits, and conclusion aligns with the overall theme of the blog post.

        5. **Final Review**:
           Read through the entire blog post to ensure it is cohesive, well-organized, and free of any grammatical or spelling errors. Make any necessary adjustments to improve clarity and readability.

        6. **Output the Blog Post**:
           Provide the final compiled blog post as your answer. Ensure that the output includes only the blog post text without any additional information or instructions.

        By following these steps, you will create a single cohesive blog post that effectively communicates the benefits of morning exercise.
        """.strip()
    ),
    requirements=[
        "Include a catchy title",
        "Include an introduction paragraph",
        "Include three main benefits with explanations",
        "Include a conclusion that encourages readers to start their morning exercise routine",
    ],
    user_variables={
        "BLOG_TITLE": blog_title.value,
        "INTRODUCTION": introduction.value,
        "BENEFITS": benefits.value,
        "CONCLUSION": conclusion.value,
    },
)
assert final_blog_post.value is not None, (
    'ERROR: task "final_blog_post" execution failed'
)


final_answer = final_blog_post.value

print(final_answer)
