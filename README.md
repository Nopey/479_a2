# seasat constraint solver
A constraint satisfaction solver written in Rust for my AI Course at VIU.


## Compilation
To compile, run `make`.
To make a debug build, run `make seasat_dbg`
`make clean` is also supported.

Code documentation can be generated, run `make doc` to generate the documentation from the doc comments.
Once generated, the documentation can be found in the `doc` folder, start at `doc/seasat/index.html`.

No moving of source code is needed, the directory structure is flat
(with the exception of the doc folder for documentation, which is automatically created).


## Running

To run, simply enter `./seasat`.

Optionally, an input filename can be specified.

If you'd like to enter input problems through stdin, use `-` as the filename
and enter the problem-- followed by an EOF (CTRL-D).


### Input Format
Comments are started with //, and run until the end of the line.
A file has one or more input problems, which all take the following form:

* N, the number of symbols, followed by newline.
* N unique identifiers for the symbols.
* C, the number of symbols, followed by the newline.
* Three times C identifiers (that have been declared in the previous symbol section),
    which form C constraint groups of three symbol identifiers.


### Output
The seasat agent will output each problem that it has successfully parsed (formatted into one line),
followed by a line containing the solution (or "COULD NOT SOLVE" if a solution was not found).

If there was an error parsing the input file, it will be output last
(after solving as many problems as would successfully parse).

Example Output:
```
IN:  { a, b, c, d, e, f, g }; { (c,e,f), (f,d,e), (e,a,g) }
OUT: { a=2, b=1, c=6, d=5, e=0, f=4, g=3 }

IN:  { a, b, c }; { (a,c,b), (c,a,b), (b,a,c) }
COULD NOT SOLVE

seasat: Could not parse input #3 from file 'demo-input.txt': expected 1 more constraint(s); parsed 0 before hitting EOF
```

If an error occurs (due to invalid commandline arguments, botched file IO, or malformed input) the process will exit with code 1.
Failing to solve an input problem is not considered an error.

## Bookkeeping
In order to facilitate the implementation of the heuristics (described in the next section),
my code maintains each variables' set of valid values in a vector.

The vector of valid value vectors cycles through variables `valid_values`, `old_valid_values`, and `futures`.

## Heuristics
The variable selection heuristics, minimum remaining values with most constraining variable as a tie-breaker,
are implemented at the top of Input::solve_inner (in input.rs).

The value selection heuristic, most constrained variable, is also implemented in input.rs's Input::solve_inner;
To find it, search for 'most constrained variable' or the 'sort_by_cached_key' function that it is implemented with.

The value selection heuristic is implemented by sorting the set of valid values for the chosen variable by their appeal,
before iterating through them until a solution is found by the recursive call.

The variable heuristic is implemented as a call to Iterator::min rather than the sort function, as the heuristic's
results will change between variable selection due to the shrinking sets of valid_values and the stubborn nature
of assigning values to variables.


## Bugs
While there are no currently known bugs, this code is not to be trusted.

