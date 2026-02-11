# pytest: skip_always
import openai

PORT = 8080

client = openai.OpenAI(api_key="na", base_url=f"http://0.0.0.0:{PORT}/v1")

response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Find all the real roots of x^3 + 1."}],
    model="granite3.3:8b",
)

print(response.choices[0])
