from openai import OpenAI

import constants

client = OpenAI(
    api_key=constants.OPENAI_API_KEY,
)

def build_poem(pet_expression: str | None) -> str | None:
    """Build a cute poem based on the pet's expression."""
    if not pet_expression:
        return None

    prompt = f"Згенеруй невелике хоку про {pet_expression} тваринку, але не акцентуй увагу, яка саме тварина на фото, а просто про її емоцію."

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Ти дуже відомий та знаменитий поет"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.4
    )

    return response.choices[0].message.content.strip()
