
from textwrap import dedent

EXTRACT_OBJECTS_TEMPLATE = dedent("""
Given a spatial-reasoning question about an image, list all entities mentioned.

Instructions:
- Return ONLY a Python list of strings, where each string is the name of an entity.
- Do not include any explanation or extra text.
- The output must be valid Python list syntax.

Example:
[Question] You are standing at the airplane’s position, facing where it is facing. Is the person on your left or right?
[Detect] ["airplane", "person"]

Now, for the question below, list the entities:
[Question] {question}
[Detect]
""")

PERSPECTIVE_CHANGE_TEMPLATE = dedent("""
Given a spatial-reasoning question, identify the perspective from which the question is asked.
- If the perspective is the camera, return: camera
- Otherwise, return only the entity's name (e.g., woman, airplane, dog).

Example:
[Question] From the woman’s perspective, is the tree on the left or right?
[Options] [tree, woman]
woman

Your Task:
Given the question below, specify the perspective.
Return only a single word: the entity's name or camera. Do not include any extra text or formatting.

[Question] {question}
[Options] {options}
""")

EGOCENTRIC_REPHRASING_TEMPLATE = dedent("""
Remove the perspective description from the following question.

Example:
[Question] From the car’s perspective, which is on the right side: the person or the tree?
Which is on the right side: the person or the tree?

Your Task:
Given the question below, remove the perspective description.
Return only the rephrased question, with no extra text or labels.

[Question] {question}
""")


if __name__ == "__main__":
    x = EXTRACT_OBJECTS_TEMPLATE.format(question="what is the meaning of life?")
    print(x)