import os
from collections.abc import AsyncGenerator

from dotenv import load_dotenv
from google import genai
from tac import TAC, TACConfig
from tac.channels.voice import VoiceChannel
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.server import TACFastAPIServer

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
AGENT_NAME = os.getenv("AGENT_NAME", "Ava")
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
print(f"Using Gemini model: {MODEL}")

SYSTEM_PROMPT = f"""
You are {AGENT_NAME}, a helpful voice AI agent on a live phone call.
Speak in short, natural sentences. Ask one question at a time.
Use lookup_customer when the caller asks about an account.
Ask for their customer ID if they have not provided it.
Do not use markdown, bullet points, links, or code blocks.
""".strip()

tac = TAC(config=TACConfig.from_env())
assert not tac.is_orchestrator_enabled(), (
    "This workshop expects relay-only mode — "
    "unset TWILIO_CONVERSATION_CONFIGURATION_ID."
)
voice_channel = VoiceChannel(tac)

CUSTOMERS = {
    "CUST-1001": {
        "name": "Sam Rivera",
        "plan": "Pro",
        "account_status": "active",
    }
}


def lookup_customer(customer_id: str) -> dict[str, str]:
    """Look up a customer account by customer ID."""
    customer = CUSTOMERS.get(customer_id.upper())
    if customer is None:
        return {"status": "not_found", "customer_id": customer_id}
    return {"status": "found", **customer}


sessions = {}


async def handle_message_ready(
    user_message: str,
    context: ConversationSession,
    memory_response: TACMemoryResponse | None,
) -> None:
    conv_id = context.conversation_id

    if conv_id not in sessions:
        sessions[conv_id] = client.aio.chats.create(
            model=MODEL,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "tools": [lookup_customer],
            },
        )

    chat = sessions[conv_id]

    async def stream_tokens() -> AsyncGenerator[str, None]:
        last_error = None
        for attempt in range(3):
            try:
                async for chunk in await chat.send_message_stream(user_message):
                    if chunk.text:
                        yield chunk.text
                return
            except Exception as error:
                last_error = error
                print(f"Gemini error (attempt {attempt + 1}/3):", error)
                if "503" not in str(error) and "UNAVAILABLE" not in str(error):
                    break
        yield "I had trouble thinking through that. Could you repeat it?"
        if last_error:
            print("Gemini gave up after retries:", last_error)

    await voice_channel.send_response(conv_id, stream_tokens())


tac.on_message_ready(handle_message_ready)


async def handle_conversation_ended(context: ConversationSession) -> None:
    sessions.pop(context.conversation_id, None)


tac.on_conversation_ended(handle_conversation_ended)

if __name__ == "__main__":
    server = TACFastAPIServer(tac=tac, voice_channel=voice_channel)
    server.start()
