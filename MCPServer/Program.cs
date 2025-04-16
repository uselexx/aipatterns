
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
    .WithTools<EchoTool>();

var app = builder.Build();

app.MapMcp();

app.Run("http://localhost:5000");

[McpServerToolType]
public class EchoTool
{
    [McpServerTool, Description("Echoes the message back to the client.")]
    public static string Echo(string message) => $"hello {message}";
}

[McpServerToolType]
public class CampusTool
{
    [McpServerTool, Description("Shows all blogs from the campus.")]
    public static string ListBlogs() => "Test blog";
}

