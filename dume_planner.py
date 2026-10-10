
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3:latest"


SYSTEM_PROMPT = """
You are DUDE, a helpful robot assistant's command planner.

Convert the user's request into valid JSON with these fields:
- goal: short description of the goal
- target_object: the object the user wants, as a simple noun phrase
- action: "find", "retrieve", "describe", or "other"
- needs_clarification: true or false
- response: a short explanation

Rules:
1. "Find my phone" means target_object = "phone".
2. "Find my charger" means target_object = "charger".
3. "Find my mobile" means target_object = "mobile phone".
4. "Find a bottle" means target_object = "bottle".
5. Words such as "my", "the", and "a" do not make a request ambiguous.
6. Do not ask which specific phone or charger the user means
   unless they explicitly refer to multiple objects.
7. For ordinary requests to find an object, set action = "find".
8. Set needs_clarification = false when the object is identifiable.
9. Never claim an object was visually detected. The vision system
   will determine that separately.
10. Never claim the robot physically moved or picked anything up.
11. Ask for clarification only when the requested object itself
    cannot reasonably be identified.
12. Return valid JSON only, with no Markdown fences.
"""


def plan_command(command):
    payload = {
        "model": MODEL,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": command}
        ],
        "options": {"temperature": 0}
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )
    response.raise_for_status()

    content = response.json()["message"]["content"]
    return json.loads(content)


if __name__ == "__main__":
    print("DUM-E language planner ready.")
    print("Type 'exit' to stop.")

    while True:
        command = input("\nYou: ").strip()

        if command.lower() == "exit":
            break

        try:
            plan = plan_command(command)
            print("\nDUM-E's plan:")
            print(json.dumps(plan, indent=2))

        except requests.RequestException as error:
            print("Could not connect to Ollama:", error)
        except (json.JSONDecodeError, KeyError) as error:
            print("Could not parse the model response:", error)