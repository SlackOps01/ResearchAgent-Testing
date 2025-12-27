import asyncio
from agents.manager import AgentManager

async def main():
    manager = AgentManager()
    message_history = []
    
    print("=== Multi-Agent System (Ollama) ===")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        try:
            prompt = input("\nUser: ")
            if prompt.lower() in ["exit", "quit"]:
                break
            
            if not prompt.strip():
                continue

            print("\nThinking...")
            response = await manager.handle_request(prompt, message_history=message_history)
            
            # Update history
            message_history = response.all_messages()
            
            print(f"\nAssistant: {response.output}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
