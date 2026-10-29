Create a tool that fetches "Stock Prices." 

However, the underlying data is "messy" (e.g., strings like "$150.00" instead of integers). 

The Challenge: to build an agent that, when it receives a formatting error from the tool, must self-correct the input and try again without human intervention.



agent
tool fetch_stock_prices


Build a Agent loop leveraging Langchain + Openai API with state preservation and a tool for fetching the generated stock data.
Make sure the Agent has an observing the executed tool response.
Build a tool fetch stock prices that would fetch data from synthetic_messy_stock_prices. Tool should produce formating error if the value of the stock is string instead of intiger. Enfore it through Pydantic model schema and propagate error to the agent loop.




