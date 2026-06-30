from langchain_core.prompts import PromptTemplate

tempelate = PromptTemplate(
    tempelate="""
Pleasse Summaried the searched paper titled "{paper_input}"with the following specification:
Explanation Dtyle:{style_input}
Explanation Length:{length_input}
1. Matematical Details:
   - Include relevant mathematical equations if present in the paper.
   - Explain the mathematical concepts using simple ,Intutive code snipets where available
2. Analogies:
   - Use relateable analogies to simplify complex ideas.
If certain information is not avaialble in the paper, respond with "Insufficient
information avaiable" Instead of guessing.
Ensure that the summary is clear, accurate and alligned with the provided style and lenght.
""",
input_variables=["paper_input","style_input","lenght_input"]
)

tempelate.save("Tempelate.json")
