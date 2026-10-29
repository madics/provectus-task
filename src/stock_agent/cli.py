from dotenv import load_dotenv

from .agent import StockAgent, configured_model, default_data_path


def interact(agent, read=input, emit=print):
    emit("Stock-price agent. /new starts an isolated conversation; /exit quits.")
    while True:
        try:
            request = read("> ").strip()
        except (EOFError, KeyboardInterrupt):
            emit("Goodbye.")
            return
        if request == "/exit":
            return
        if request == "/new":
            agent.new_conversation()
            emit(f"New conversation: {agent.thread_id}")
        elif request:
            agent.run(request, emit=emit)


def main():
    load_dotenv()
    try:
        agent = StockAgent(configured_model(), default_data_path())
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    interact(agent)


if __name__ == "__main__":
    main()
