# pytest: skip_always

import textwrap

from validations.val_fn_2 import validate_input as val_fn_2

import mellea
from mellea.stdlib.requirements.requirement import req

m = mellea.start_session()


# 1. Brainstorm and select a catchy title related to the benefits of morning exercise. - - BLOG_TITLE
blog_title = m.instruct(
    textwrap.dedent(
        R"""
        Your task is to brainstorm and select a catchy title related to the benefits of morning exercise for a blog post. Follow these steps to accomplish your task:

        1. **Understand the Topic**:
           Begin by reflecting on the topic, which is the benefits of morning exercise. Consider what makes morning workouts unique and advantageous.

        2. **Brainstorm Title Ideas**:
           Generate several title options that are catchy, engaging, and clearly indicate the content of your blog post. The titles should pique readers' interest and accurately represent the topic.

        3. **Select the Best Title**:
           From your brainstormed list, choose the most compelling and relevant title for a blog post about the benefits of morning exercise. Ensure it is concise, informative, and appealing to your target audience.

        4. **Finalize Your Choice**:
           Once you've selected the best title, prepare to use it as the basis for your blog post. This chosen title will be referenced in subsequent steps for compiling the complete blog post content.

        Remember, your goal is to create a title that effectively communicates the value of morning exercise and encourages readers to continue reading the blog post.
        """.strip()
    ),
    requirements=[
        "The blog post must have a catchy title",
        req(
            "The post should include an introduction paragraph", validation_fn=val_fn_2
        ),
        "Three main benefits of morning exercise with explanations are required",
        "A conclusion that encourages readers to start their morning exercise routine is necessary",
    ],
)
assert blog_title.value is not None, 'ERROR: task "blog_title" execution failed'

# 2. Write an introduction paragraph that engages readers and introduces the topic of morning exercise. - - INTRODUCTION
introduction = m.instruct(
    textwrap.dedent(
        R"""
        Your task is to write an introduction paragraph for a blog post about the benefits of morning exercise. This paragraph should be engaging, capture the reader's attention, and introduce the topic effectively. Follow these steps:

        1. **Understand the Topic**:
           Begin by reviewing the overall theme of the blog post, which focuses on the benefits of morning exercise. Keep this in mind as you craft your introduction.

        2. **Engage Readers**:
           Start with a hook to draw readers into the topic. This could be an intriguing question, a surprising fact, or a relatable scenario related to mornings and exercise.

        3. **Introduce Morning Exercise**:
           After capturing attention, briefly introduce morning exercise as the subject of your blog post. Mention that you will explore its benefits in detail throughout the article.

        4. **Set Expectations**:
           Conclude the introduction by hinting at what readers can expect to learn from the rest of the post. This should create anticipation and encourage them to continue reading.

        Write a concise yet compelling introduction paragraph (approximately 5-7 sentences) that adheres to these guidelines, ensuring it sets the stage for the detailed exploration of morning exercise benefits in subsequent sections of your blog post.
        """.strip()
    ),
    requirements=None,
    user_variables={},
)
assert introduction.value is not None, 'ERROR: task "introduction" execution failed'

# 3. Identify three main benefits of morning exercise, then write detailed explanations for each benefit. - - BENEFITS_EXPLANATION
benefits_explanation = m.instruct(
    textwrap.dedent(
        R"""
        Your task is to identify three main benefits of morning exercise and provide detailed explanations for each. Follow these steps to accomplish your task:

        1. **Research Morning Exercise Benefits**: Begin by researching the advantages associated with morning workouts. Focus on finding credible sources that highlight key benefits. Some common benefits include improved mood, increased energy levels throughout the day, and better sleep quality.

        2. **Select Three Key Benefits**: From your research, choose three of the most significant benefits to focus on for this blog post. Ensure these benefits are well-supported by evidence from reliable sources.

        3. **Write Detailed Explanations**: For each selected benefit, write a detailed explanation. Each explanation should include:
           - A clear statement of the benefit.
           - Scientific or empirical evidence supporting the benefit.
           - Practical examples or anecdotes to illustrate how this benefit manifests in real life.

        4. **Format for Blog Post**: Structure your explanations in a way that they can be easily integrated into the blog post format. Consider using bullet points or numbered lists for clarity if needed.

        5. **Compile Explanations**: Compile these detailed explanations into a structured format, ready to be combined with other parts of the blog post (title and conclusion) in the next steps.

        Ensure that your explanations are clear, informative, and engaging, maintaining a tone suitable for a health and wellness blog audience.
        """.strip()
    ),
    requirements=None,
    user_variables={},
)
assert benefits_explanation.value is not None, (
    'ERROR: task "benefits_explanation" execution failed'
)

# 4. Compile the identified benefits with their explanations into a structured format for the blog post. - - STRUCTURED_BENEFITS
structured_benefits = m.instruct(
    textwrap.dedent(
        R"""
        Your task is to compile the identified benefits of morning exercise and their detailed explanations into a structured format suitable for a blog post. Follow these steps to accomplish your task:

        First, review the benefits of morning exercise that have been identified and explained in the previous step:
        <benefits_explanation>
        {{BENEFITS_EXPLANATION}}
        </benefits_explanation>

        Next, organize these benefits into a clear and logical structure for the blog post. A common format for presenting benefits is to list each benefit as a distinct point or heading, followed by an explanation of that benefit.

        For example, you might structure your compiled benefits like this:

        ### Benefit 1: [Benefit Name]
        [Detailed Explanation of the benefit]

        ### Benefit 2: [Benefit Name]
        [Detailed Explanation of the benefit]

        ### Benefit 3: [Benefit Name]
        [Detailed Explanation of the benefit]

        Ensure that each benefit is presented clearly and concisely, with a heading that accurately reflects the nature of the benefit. The explanation should be comprehensive yet easy to understand, providing readers with valuable insights into why morning exercise is beneficial.

        Finally, ensure this structured format aligns with the overall flow and style of the blog post being created. This compiled version will be used in conjunction with other parts of the blog post (title, introduction, conclusion) to form a cohesive piece of content.
        """.strip()
    ),
    requirements=[
        "The blog post must have a catchy title",
        req(
            "The post should include an introduction paragraph", validation_fn=val_fn_2
        ),
        "Three main benefits of morning exercise with explanations are required",
        "A conclusion that encourages readers to start their morning exercise routine is necessary",
    ],
    user_variables={"BENEFITS_EXPLANATION": benefits_explanation.value},
)
assert structured_benefits.value is not None, (
    'ERROR: task "structured_benefits" execution failed'
)

# 5. Write a conclusion paragraph that encourages readers to start incorporating morning exercise into their routine. - - CONCLUSION
conclusion = m.instruct(
    textwrap.dedent(
        R"""
        Your task is to write a conclusion paragraph that encourages readers to start incorporating morning exercise into their routine for the blog post about the benefits of morning exercise. Follow these steps:

        1. **Review Previous Steps**:
           Begin by reviewing the structured benefits and explanations from previous steps, which can be found here:
           <structured_benefits>
           {{STRUCTURED_BENEFITS}}
           </structured_benefits>

           This will help you understand the key points that have already been covered in the blog post.

        2. **Summarize Key Benefits**:
           Briefly summarize the main benefits of morning exercise as discussed earlier to remind readers of their significance. You can refer to {{BENEFITS_EXPLANATION}} for specifics on each benefit.

        3. **Craft an Encouraging Message**:
           Write a paragraph that motivates readers to start their own morning exercise routine. Use persuasive language and emphasize the positive impact of making this change.

        4. **Structure the Conclusion**:
           Ensure your conclusion is concise, engaging, and effectively encourages action from the reader. It should wrap up the blog post by reinforcing the importance of morning exercise and inspiring readers to take the first step towards integrating it into their daily routine.

        Here's a suggested structure for your conclusion paragraph:
           - Briefly restate or reference one or two key benefits discussed in the body of the blog post.
           - Use motivational language to encourage readers to adopt morning exercise.
           - Provide a clear, actionable call to action (e.g., "Start today with a short walk around your neighborhood").

        5. **Finalize and Output**:
           Once you have written the conclusion paragraph, ensure it is polished and ready for integration into the final blog post. Do not include any additional information or explanation beyond the requested conclusion text.
        """.strip()
    ),
    requirements=[
        "The blog post must have a catchy title",
        req(
            "The post should include an introduction paragraph", validation_fn=val_fn_2
        ),
        "Three main benefits of morning exercise with explanations are required",
        "A conclusion that encourages readers to start their morning exercise routine is necessary",
    ],
    user_variables={
        "STRUCTURED_BENEFITS": structured_benefits.value,
        "BENEFITS_EXPLANATION": benefits_explanation.value,
    },
)
assert conclusion.value is not None, 'ERROR: task "conclusion" execution failed'

# 6. Combine all parts (title, introduction, benefits explanation, and conclusion) into a cohesive short blog post. - - FINAL_BLOG_POST
final_blog_post = m.instruct(
    textwrap.dedent(
        R"""
        Your task is to combine all the identified parts—the catchy title, engaging introduction, detailed explanations of three main benefits, and an encouraging conclusion—into a cohesive short blog post about the benefits of morning exercise. Follow these steps to accomplish your task:

        First, review the components you have already generated for this subtask:
        <generated_components>
        {{BLOG_TITLE}}
        {{INTRODUCTION}}
        {{BENEFITS_EXPLANATION}}
        {{STRUCTURED_BENEFITS}}
        {{CONCLUSION}}
        </generated_components>

        Next, structure the blog post by arranging these components in a logical order. A typical structure for a short blog post would be:
        1. **Title**: Start with your catchy title related to morning exercise benefits.
        2. **Introduction**: Follow the title with an engaging introduction paragraph that sets the stage and introduces the topic of morning exercise.
        3. **Main Benefits Section**: Present the three main benefits of morning exercise, each accompanied by a detailed explanation as you have generated in {{BENEFITS_EXPLANATION}}. Ensure these explanations are clear, concise, and supported by the structured information in {{STRUCTURED_BENEFITS}}.
        4. **Conclusion**: End with an encouraging conclusion that motivates readers to start or continue their morning exercise routine. Use the provided {{CONCLUSION}} for this purpose.

        Write a cohesive blog post by weaving these components together, ensuring smooth transitions between sections. Maintain a friendly and informative tone throughout the post, keeping it concise yet comprehensive.

        Finally, present only the completed short blog post as your answer without any additional information or explanation.
        """.strip()
    ),
    requirements=[
        "The blog post must have a catchy title",
        req(
            "The post should include an introduction paragraph", validation_fn=val_fn_2
        ),
        "Three main benefits of morning exercise with explanations are required",
        "A conclusion that encourages readers to start their morning exercise routine is necessary",
    ],
    user_variables={
        "BLOG_TITLE": blog_title.value,
        "INTRODUCTION": introduction.value,
        "BENEFITS_EXPLANATION": benefits_explanation.value,
        "STRUCTURED_BENEFITS": structured_benefits.value,
        "CONCLUSION": conclusion.value,
    },
)
assert final_blog_post.value is not None, (
    'ERROR: task "final_blog_post" execution failed'
)


final_answer = final_blog_post.value

print(final_answer)
