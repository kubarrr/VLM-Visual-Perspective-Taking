
from textwrap import dedent

EXTRACT_OBJECTS_TEMPLATE = dedent("""
Given a spatial-reasoning question about an image, list all entities mentioned.

Example:
[Question] You are standing at the airplane’s position, facing where it is facing. Is the person on your left or right?
[Detect] [airplane, person]

Now, for the question below, list the entities:
[Question] {question}
[Detect]
""")

PERSPECTIVE_CHANGE_TEMPLATE = dedent("""
Given a question about spatial reasoning, we want to extract the perspective of the
question. If the question is from the camera’s perspective, return ++camera++.
# Example
[Question] From the woman’s perspective, is the tree on the left or right?
[Perspective] ++woman++
# Your Task
Given the question below, please specify  the perspective from which the question is asked.
You must return in the format:
[Perspective] ++object name++
[Question] {question}
[Options] {options} camera
[Perspective]
""")

EGOCENTRIC_REPHRASING_TEMPATE = dedent("""
From a sentence with a perspective description, we need to remove the perspective description.
# Example
[Question] From the car’s perspective, which is on the right side: the person or the tree?
[Output] Which is on the right side: the person or the tree?
# Your Task
Given the question below, please remove the perspective description.
""")


if __name__ == "__main__":
    x = EXTRACT_OBJECTS_TEMPLATE.format(question="what is the meaning of life?")
    print(x)