## ReACT TURN NUMBER 1
### Thought
I need to first find out the ZIP code for Thornton, NH using the Zip Code Lookup tool, and then use that ZIP code with the Get the weather tool to determine today's high temperature.
### Action
Zip Code Lookup
### Arguments for action
```json
{
  "city": "Thornton, NH"
}
```
### Observation
03285
### Done Check
Not done.
## ReACT TURN NUMBER 2
### Thought
I now need to use the Get the weather tool with the ZIP code 03285 for Thornton, NH to get today's high temperature.
### Action
Get the weather
### Arguments for action
```json
{
  "zip_code": "03285"
}
```
### Observation
The weather in Thornton, NH is sunny with a high of 78 and a low of 52. Scattered showers are possible in the afternoon.
### Done Check
Done. Will summarize and return output now.
## Final Answer
Today's high temperature in Thornton, NH is 78 degrees.
