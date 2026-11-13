from dataclasses import dataclass
from drafter import *
from bakery import assert_equal
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class GraphRecord:
    """Information about one equation saved in the user's history."""
    equation: str
    x_min: float
    x_max: float


@dataclass
class State:
    """Information the website remembers while the user visits pages."""
    graphs: list[GraphRecord]
    x_min: float
    x_max: float
    graph_count: int
    last_equation: str


def split_equation(equation: str) -> list[str]:
    """Separate the two sides of an equation, if it has an equals sign."""
    equation = equation.lower().replace(" ", "")
    if "=" not in equation:
        return [equation]
    parts = equation.split("=")
    if len(parts) != 2 or parts[0] == "" or parts[1] == "":
        return []
    return parts


def tokenize_equation(equation: str) -> list[str]:
    """Break a math expression into numbers, variables, and operators."""
    equation = equation.lower().replace(" ", "")
    tokens = []
    position = 0

    while position < len(equation):
        character = equation[position]

        if character.isdigit() or character == ".":
            number = ""
            decimal_count = 0
            while position < len(equation):
                character = equation[position]
                if character.isdigit():
                    number += character
                    position += 1
                elif character == "." and decimal_count == 0:
                    number += character
                    decimal_count += 1
                    position += 1
                else:
                    break
            if number == ".":
                return []
            tokens.append(number)

        elif character in "xy":
            tokens.append(character)
            position += 1

        elif character in "+-*/^()":
            if character == "*" and position + 1 < len(equation):
                if equation[position + 1] == "*":
                    tokens.append("^")
                    position += 2
                else:
                    tokens.append(character)
                    position += 1
            else:
                tokens.append(character)
                position += 1
        else:
            return []

    # Add multiplication for examples such as 2x, 3y, and 2(x + y).
    expanded_tokens = []
    for token in tokens:
        if expanded_tokens:
            previous = expanded_tokens[-1]
            previous_ends_value = (
                previous in ["x", "y", ")"]
                or previous[0].isdigit()
                or previous[0] == "."
            )
            current_starts_value = (
                token in ["x", "y", "("]
                or token[0].isdigit()
                or token[0] == "."
            )
            if previous_ends_value and current_starts_value:
                expanded_tokens.append("*")
        expanded_tokens.append(token)

    return expanded_tokens


def parse_primary(
    tokens: list[str], position: list[int], x: float, y: float
) -> float | None:
    """Read a number, x, y, or a parenthesized expression."""
    if position[0] >= len(tokens):
        return None

    token = tokens[position[0]]

    if token == "(":
        position[0] += 1
        value = parse_expression(tokens, position, x, y)
        if value is None or position[0] >= len(tokens):
            return None
        if tokens[position[0]] != ")":
            return None
        position[0] += 1
        return value

    if token == "x":
        position[0] += 1
        return x

    if token == "y":
        position[0] += 1
        return y

    if token[0].isdigit() or token[0] == ".":
        position[0] += 1
        return float(token)

    return None


def parse_power(
    tokens: list[str], position: list[int], x: float, y: float
) -> float | None:
    """Read a value and an optional exponent."""
    value = parse_primary(tokens, position, x, y)
    if value is None:
        return None
    if position[0] < len(tokens) and tokens[position[0]] == "^":
        position[0] += 1
        exponent = parse_unary(tokens, position, x, y)
        if exponent is None or abs(exponent) > 10:
            return None
        value = value ** exponent
    return value


def parse_unary(
    tokens: list[str], position: list[int], x: float, y: float
) -> float | None:
    """Handle positive and negative values."""
    if position[0] < len(tokens) and tokens[position[0]] == "+":
        position[0] += 1
        return parse_unary(tokens, position, x, y)
    if position[0] < len(tokens) and tokens[position[0]] == "-":
        position[0] += 1
        value = parse_unary(tokens, position, x, y)
        if value is None:
            return None
        return -value
    return parse_power(tokens, position, x, y)


def parse_term(
    tokens: list[str], position: list[int], x: float, y: float
) -> float | None:
    """Handle multiplication and division."""
    value = parse_unary(tokens, position, x, y)
    if value is None:
        return None
    while position[0] < len(tokens):
        operator = tokens[position[0]]
        if operator != "*" and operator != "/":
            break
        position[0] += 1
        next_value = parse_unary(tokens, position, x, y)
        if next_value is None:
            return None
        if operator == "*":
            value = value * next_value
        else:
            if next_value == 0:
                return None
            value = value / next_value
    return value


def parse_expression(
    tokens: list[str], position: list[int], x: float, y: float
) -> float | None:
    """Handle addition and subtraction."""
    value = parse_term(tokens, position, x, y)
    if value is None:
        return None
    while position[0] < len(tokens):
        operator = tokens[position[0]]
        if operator != "+" and operator != "-":
            break
        position[0] += 1
        next_value = parse_term(tokens, position, x, y)
        if next_value is None:
            return None
        if operator == "+":
            value = value + next_value
        else:
            value = value - next_value
    return value


def evaluate_expression(expression: str, x: float, y: float) -> float | None:
    """Evaluate an expression containing x and/or y without using eval."""
    tokens = tokenize_equation(expression)
    if not tokens:
        return None
    position = [0]
    value = parse_expression(tokens, position, x, y)
    if position[0] != len(tokens):
        return None
    return value


def calculate_y(equation: str, x: float) -> float | None:
    """Calculate y for a function equation such as y = 2x + 1."""
    parts = split_equation(equation)
    if len(parts) == 0:
        return None
    if len(parts) == 1:
        expression = parts[0]
    elif parts[0] == "y":
        expression = parts[1]
    else:
        return None
    return evaluate_expression(expression, x, 0)


def calculate_relation(equation: str, x: float, y: float) -> float | None:
    """Calculate left side minus right side for an implicit equation."""
    parts = split_equation(equation)
    if len(parts) != 2:
        return None
    left = evaluate_expression(parts[0], x, y)
    right = evaluate_expression(parts[1], x, y)
    if left is None or right is None:
        return None
    return left - right


def make_x_values(x_min: float, x_max: float) -> list[float]:
    """Create 101 evenly spaced x values for the graph."""
    x_values = []
    step = (x_max - x_min) / 100
    for position in range(101):
        x_values.append(x_min + position * step)
    return x_values


# Tests for the equation calculator.
assert_equal(calculate_y("2*x + 1", 3), 7)
assert_equal(calculate_y("x^2 - 4", 3), 5)
assert_equal(calculate_y("2x + 1", 3), 7)
assert_equal(calculate_y("(x + 2) * 3", 1), 9)
assert_equal(calculate_y("-x^2", 3), -9)
assert_equal(calculate_y("x / 0", 2), None)
assert_equal(len(make_x_values(-5, 5)), 101)
assert_equal(calculate_relation("x^2 + y^2 = 25", 3, 4), 0)
assert_equal(calculate_relation("x + y = 10", 2, 3), -5)
assert_equal(calculate_y("y = 2x + 1", 3), 7)



@route
def index(state: State) -> Page:
    """Show a welcoming home page with clear navigation."""
    return Page(state, [
        Header("Graphing Calculator", 1),
        Text("Explore equations and see their graphs come to life."),
        HorizontalRule(),
        Div(
            Header("Start Here", 2),
            Text("Create a graph from a function or a two-variable equation."),
            Button("Open Graph Calculator", "graph_calculator"),
        ),
        Div(
            Header("Explore", 2),
            Row(
                Button("Graph History", "graph_history"),
                Button("How to Use", "help_page"),
            ),
        ),    
    ])


@route
def graph_calculator(state: State) -> Page:
    """Show the equation and graph-range form."""
    return Page(state, [
        Header("Create a Graph", 1),
        Text("Enter an equation, choose the viewing range, and graph it."),
        HorizontalRule(),
        Div(
            Header("Equation", 2),
            Text("Function example: y = 2x + 1"),
            Text("Circle example: x^2 + y^2 = 25"),
            TextBox("equation", state.last_equation),
        ),
        Div(
            Header("Viewing Window", 2),
            Text("Minimum x and y value:"),
            TextBox("x_min", str(state.x_min)),
            Text("Maximum x and y value:"),
            TextBox("x_max", str(state.x_max)),
        ),
        Button("Graph Equation", "create_graph"),
        Button("Back Home", "index"),
        HorizontalRule(),
        Text("Tip: Use ^ for exponents, such as x^2. Parentheses are supported."),
    ])


@route
def create_graph(
    state: State,
    equation: str,
    x_min: float,
    x_max: float,
) -> Page:
    """Check the equation, save it, and display its graph."""
    if x_min >= x_max:
        return Page(state, [
            Header("Check Your Viewing Window", 1),
            Text("The minimum value must be smaller than the maximum value."),
            Button("Back to Calculator", "graph_calculator"),
        ])

    # Decide whether this is a function or an implicit equation.
    parts = split_equation(equation)
    is_implicit = len(parts) == 2 and parts[0] != "y"

    valid_equation = len(parts) > 0
    if is_implicit:
        if calculate_relation(equation, 1.0, 1.0) is None:
            valid_equation = False
    else:
        test_values = [-2.0, 0.0, 2.0]
        for test_x in test_values:
            if calculate_y(equation, test_x) is None:
                valid_equation = False

    if not valid_equation:
        return Page(state, [
            Header("Equation Not Recognized", 1),
            Text("Check the equation and try again."),
            Text("Function: y = 2x + 1"),
            Text("Circle: x^2 + y^2 = 25"),
            Button("Try Again", "graph_calculator"),
        ])

    record = GraphRecord(equation, x_min, x_max)
    state.graphs.append(record)
    state.x_min = x_min
    state.x_max = x_max
    state.graph_count += 1
    state.last_equation = equation

    return graph_results(state)


@route
def graph_results(state: State) -> Page:
    """Draw either a function graph or an implicit curve."""
    if not state.graphs:
        return Page(state, [
            Header("No Graph Yet", 1),
            Text("Create a graph first using the calculator."),
            Button("Open Calculator", "graph_calculator"),
        ])

    graph = state.graphs[-1]
    parts = split_equation(graph.equation)
    is_implicit = len(parts) == 2 and parts[0] != "y"

    plt.figure(figsize=(7, 5))

    if is_implicit:
        # Make a grid of x and y values, then draw where left - right = 0.
        x_values = np.linspace(graph.x_min, graph.x_max, 101)
        y_values = np.linspace(graph.x_min, graph.x_max, 101)
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        z_values = np.zeros((101, 101))

        for row in range(101):
            for column in range(101):
                result = calculate_relation(
                    graph.equation,
                    float(x_grid[row, column]),
                    float(y_grid[row, column])
                )
                if result is None or abs(result) >= 1000000:
                    z_values[row, column] = np.nan
                else:
                    z_values[row, column] = result

        plt.contour(x_grid, y_grid, z_values, levels=[0])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(graph.equation)
    else:
        x_values = make_x_values(graph.x_min, graph.x_max)
        y_values = []
        plotted_x_values = []

        for x in x_values:
            y = calculate_y(graph.equation, x)
            if y is not None and abs(y) < 1000000:
                plotted_x_values.append(x)
                y_values.append(y)

        plt.plot(plotted_x_values, y_values)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(graph.equation)

    plt.axhline(y=0)
    plt.axvline(x=0)
    plt.grid(True)

    return Page(state, [
        Header("Your Graph", 1),
        Div(
            Header("Equation", 2),
            Text(graph.equation),
            Text("Viewing range: " + str(graph.x_min) + " to " + str(graph.x_max)),
        ),
        MatPlotLibPlot(),
        Text("Graphs created: " + str(state.graph_count)),
        Row(
            Button("Graph Another Equation", "graph_calculator"),
            Button("View History", "graph_history"),
        ),
        Button("Home", "index"),
    ])


@route
def graph_history(state: State) -> Page:
    """List the equations saved during this visit."""
    content = [
        Header("Graph History", 1),
        Text("Graphs created: " + str(state.graph_count)),
        HorizontalRule(),
    ]

    if not state.graphs:
        content.append(Text("No graphs saved yet. Create your first graph!"))
    else:
        for position, graph in enumerate(state.graphs):
            content.append(
                Div(
                    Header("Graph " + str(position + 1), 3),
                    Text(graph.equation),
                    Text(
                        "Range: " + str(graph.x_min) + " to " + str(graph.x_max)
                    ),
                    Button(
                        "View Details",
                        "graph_details",
                        Argument("graph_index", position)
                    ),
                )
            )

    content.append(HorizontalRule())
    content.append(Row(
        Button("Clear History", "clear_history"),
        Button("Back to Calculator", "graph_calculator"),
    ))
    content.append(Button("Home", "index"))
    return Page(state, content)


@route
def graph_details(state: State, graph_index: int) -> Page:
    """Show the saved settings for one graph."""
    if graph_index < 0 or graph_index >= len(state.graphs):
        return Page(state, [
            Header("Graph Not Found", 1),
            Text("That graph is not in your history."),
            Button("Return to History", "graph_history"),
        ])

    graph = state.graphs[graph_index]
    return Page(state, [
        Header("Graph Details", 1),
        Div(
            Header("Equation", 2),
            Text(graph.equation),
        ),
        Div(
            Header("Viewing Window", 2),
            Text("Minimum: " + str(graph.x_min)),
            Text("Maximum: " + str(graph.x_max)),
        ),
        Row(
            Button("Return to History", "graph_history"),
            Button("Create Another Graph", "graph_calculator"),
        ),
    ])


@route
def clear_history(state: State) -> Page:
    """Remove saved graphs and reset the graph counter."""
    state.graphs = []
    state.graph_count = 0
    return Page(state, [
        Header("History Cleared", 1),
        Text("Your saved graphs have been removed."),
        Button("Create a New Graph", "graph_calculator"),
        Button("Home", "index"),
    ])


@route
def help_page(state: State) -> Page:
    """Explain how to enter functions and implicit equations."""
    return Page(state, [
        Header("How to Use the Calculator", 1),
        Text("Follow these steps to make a graph:"),
        Div(
            Header("1. Enter an equation", 2),
            Text("For a function, enter y = 2x + 1."),
            Text("For a circle, enter x^2 + y^2 = 25."),
        ),
        Div(
            Header("2. Choose the viewing window", 2),
            Text("Set the minimum and maximum values for both axes."),
        ),
        Div(
            Header("3. Create your graph", 2),
            Text("Select Graph Equation to see the result."),
        ),
        HorizontalRule(),
        Header("More Examples", 2),
        BulletedList([
            "y = x^2 - 4",
            "y = 2x^3 - 3x + 1",
            "x^2 + y^2 = 25",
            "x^2/9 + y^2/4 = 1",
            "x = y^2",
        ]),
        Text("Use ^ for powers. Parentheses and basic arithmetic are supported."),
        Button("Try It Now", "graph_calculator"),
        Button("Home", "index"),
    ])


set_website_title("Graphing Calculator")
set_site_information(
    author="Chris Marucci",
    description="A simple python graphing calculator",
    sources="Matplotlib, Drafter",
    planning=["CS1 Final Project 1_ Design.pdf"],
    links=["https://github.com/Chris-Marucci/Graphing-Calc"]
)
hide_debug_information()
set_website_framed(False)

start_server(State([], -10.0, 10.0, 0, "y = 2x + 1"))

