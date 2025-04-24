
using ModelContextProtocol.Server;
using System.ComponentModel;

var builder = WebApplication.CreateBuilder(args);
builder.Logging.AddConsole(consoleLogOptions =>
{
    // Configure all logs to go to stderr
    consoleLogOptions.LogToStandardErrorThreshold = LogLevel.Trace;
});
builder.Services
    .AddMcpServer()
    .WithHttpTransport()
    .WithTools<CampusTool>()
    .WithTools<WeatherTool>();

var app = builder.Build();

app.MapMcp();

app.Run("http://localhost:5000");

[McpServerToolType]
public class WeatherTool
{
    [McpServerTool, Description("Provides the current weather for the given city.")]
    public static string GetWeather(string message) => $"Current weather in {message} is sunny";
}

[McpServerToolType]
public class CampusTool
{
    [McpServerTool, Description("Shows all blogs from the campus.")]
    public static string ListBlogs() => "Test blog";
}

