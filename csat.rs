//! Constraint satisfaction problem solver
//! 
//! CSat Structures, Parser, and Solver.
use std::{fmt, io, str, iter};
use std::num::ParseIntError;
use std::io::BufRead;

/// Constraint `c` is true for a vector of values `v` iff
/// * v[c[0]] < v[c[1]] < v[c[2]],
/// * v[c[1]] < v[c[2]] < v[c[0]], or
/// * v[c[2]] < v[c[0]] < v[c[1]].
#[derive(Debug, Copy, Clone, Default)]
struct Constraint([usize; 3]);

/// A solution to a constraint satisfaction problem
#[derive(Debug, Clone)]
pub struct Output {
    symbols: Vec<String>,
    values: Vec<usize>
}

impl fmt::Display for Output {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let (Some(symbol), Some(value)) = (self.symbols.first(), self.values.first()) {
            write!(f, "{{ {}={}", symbol, value)?;
            for (symbol, value) in self.symbols.iter().zip(self.values.iter()).skip(1) {
                write!(f, ", {}={}", symbol, value)?;
            }
            write!(f, " }}")
        } else {
            // U+2205 is the empty set symbol's unicode codepoint.
            // write!(f, "\u{2205}")
            // but I'd rather write the empty set as empty brackets
            write!(f, "{{}}")
        }
    }
}

/// A description of a given constraint satisfaction problem
#[derive(Debug, Clone)]
pub struct Input {
    symbols: Vec<String>,
    constraints: Vec<Constraint>
}

impl Input {
    /// Parses a single Input from the provided input,
    /// Leaving excess input in the buffer
    pub fn from_read(read: &mut dyn BufRead) -> Result<Input, InputError>
    {
        use InputError::*;

        let line = Input::get_line(read)?.ok_or(NoMoreInputs)?;
        let num_symbols = line.parse::<usize>().map_err(|e| CantParseSymbolCount(line.to_owned(), e))?;
        // println!("num_symbols: {:?}", num_symbols);

        let mut symbols = Vec::with_capacity(num_symbols);
        for i in 0..num_symbols {
            let symbol = Input::get_identifier(read)?.ok_or(ExpectedMoreSymbols(num_symbols-i, i))?;

            // Prevent duplicate declarations of symbol
            if symbols.contains(&symbol) {
                return Err(MultipleDeclarationsOfSymbol(symbol));
            }

            // println!("symbol: {:?}", symbol);

            symbols.push(symbol);
        }

        let line = Input::get_line(read)?.ok_or(ExpectedConstraintCount)?;
        let num_constraints = line.parse::<usize>().map_err(|e| CantParseConstraintCount(line.to_owned(), e))?;
        // println!("num_constraints: {:?}", num_constraints);

        let mut constraints = Vec::with_capacity(num_constraints);
        for i in 0..num_constraints {
            let mut constraint = Constraint::default();
            for x in 0..3 {
                let symbol = match Input::get_identifier(read)? {
                    Some(s) => s,
                    None if x == 0 => return Err(ExpectedMoreConstraints(num_symbols-i, i)),
                    None => return Err(PartialConstraint(x, i))
                };
                constraint.0[x] = match symbols.iter().position(|other_symbol| other_symbol == &symbol) {
                    Some(idx) => idx,
                    None => return Err(UndeclaredSymbol(symbol))
                };
            }
            // println!("constraint: {:?}", constraint);
            constraints.push(constraint);
        }

        Ok(Input{
            symbols,
            constraints
        })
    }
    /// Private helper function for reading a line from a bufread,
    /// while trimming comments, whitespace, and empty lines
    fn get_line(read: &mut dyn BufRead) -> io::Result<Option<String>>
    {
        // read_line returns 1 for empty lines, as the newlines are included
        // and so, as it says in its docs, 0 is EOF.
        let mut read_buf = String::new();
        while read.read_line(&mut read_buf)? != 0 {
            let mut line = read_buf.as_str();

            // remove comments
            // (everything following two slashes)
            if let Some(non_comment_len) = line.find("//") {
                line = &line[0..non_comment_len];
            }

            // remove whitespace
            line = line.trim();

            // return first non-empty line
            if !line.is_empty() {
                return Ok(Some(line.to_string()));
            }

            // Gotta clear read_buf because read_line appends
            read_buf.clear();
        }
        Ok(None)
    }

    /// Private helper function for reading a single identifier from a bufread.
    ///
    /// Identifiers are space delimited sequences of one or more characters
    fn get_identifier(read: &mut dyn BufRead) -> io::Result<Option<String>>
    {
        let mut identifier = String::new();
        // number of bytes we've consumed
        let mut consumed_amount = 0usize;
        // are we currently throwing out data until we hit the newline character?
        let mut in_comment = false;

        // This loop parses one identifier
        // NOTE: This 'ident: syntax is for a named break, and is not a goto.
        'ident: loop {
            // Consume character(s) from previous go through the loop
            if consumed_amount != 0 {
                read.consume(consumed_amount);
                consumed_amount = 0;
            }

            if in_comment {
                in_comment = false;
                let mut comment = String::new();
                read.read_line(&mut comment)?;

                //  if we have an identifier, it's completed by the whitespace
                if !identifier.is_empty() { break 'ident; }
            }

            // buffer of u8's (not unicode 'char's!)
            let buffer = read.fill_buf()?;
            if buffer.is_empty() { break 'ident; }

            let valid_amount = match str::from_utf8(&buffer) {
                // the whole buffer is valid UTF-8
                Ok(_) => buffer.len(),
                // there's a unicode error at the start of the buffer, so it's a real error.
                Err(e) if e.valid_up_to() == 0 =>
                    // error message akin to BufRead::read_line
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "stream did not contain valid UTF-8")),
                // there's an error, but we read something
                Err(e) => e.valid_up_to()
            };


            //
            // we read something; try to add it to the identifier
            //

            // Unwrap will never panic because the previous from_utf call says these are valid bytes
            let valid = str::from_utf8(buffer.split_at(valid_amount).0).unwrap();

            // Iterate through pairs of characters in the input,
            for (c, n) in valid.chars().zip(
                //  including the last character paired with no character following
                valid.chars().map(|c| Some(c)).chain(iter::once(None))
            ) {
                if c=='/' && n == Some('/') {
                    // Nuke until newline or EOF
                    in_comment = true;
                    consumed_amount += 2; // sizeof("//")
                    continue 'ident;
                } else if c=='/' && n == None && valid_amount != 1 {
                    // Can't consume c without next character

                    // NOTE: if valid_amount is 1, then we're at EOF, or the next character is invalid UTF:
                    //   if it's invalid UTF, we're gonna return an error
                    //   if it's EOF, we can consume c.

                    break;
                } else if c.is_whitespace() { // NOTE: newlines count for is_whitespace
                    // Consume c without appending to identifier,
                    consumed_amount += c.len_utf8();
                    //  if we have an identifier, it's completed by the whitespace
                    if !identifier.is_empty() { break 'ident; }
                } else {
                    // Consume c
                    consumed_amount += c.len_utf8();
                    //  and add it to the identifier
                    identifier.push(c);
                }
            }
        }
        // Consume character(s) from final loop
        if consumed_amount != 0 {
            read.consume(consumed_amount);
        }
        // Identifiers cannot be empty
        if identifier.is_empty() {
            return Ok(None);
        }
        // :tada:
        return Ok(Some(identifier));
    }

    /// Parses as many `Input`s as can be from input stream.
    pub fn many_from_read(read: &mut dyn BufRead) -> (Vec<Input>, InputError)
    {
        let mut inputs = vec![];
        loop {
            match Input::from_read(read) {
                Ok(input) => inputs.push(input),
                Err(err) => return (inputs, err)
            };
        }
    } 

    /// Finds subsets of the valid value sets, resulting from applying the chosen_value to chosen_var.
    /// 
    /// NOTE: garbage in, garbage out. This function does not check for the correctness of the input sets.
    fn handle_constraints(&self, chosen_var: usize, chosen_value: usize, valid_values: &mut Vec<Vec<usize>>, assigned_values: &mut [Option<usize>]){
        // Allocate a vec with size 1 for our variable
        valid_values[chosen_var] = Vec::with_capacity(1);

        // `chosen_value` has been taken, so no other variable may use it.
        for var_vals in valid_values.iter_mut() {
            if let Some(idx) = var_vals.iter().position(|&x| x == chosen_value) {
                // remove chosen_var from var_vals by swapping it with the last element
                var_vals.swap_remove(idx);
            }
            // if var_vals.len() > 0 {println!("{} !>> {:?}", chosen_value, var_vals)};
        }

        valid_values[chosen_var].push(chosen_value);

        // TODO: handle_constraints' pyramid of for loops could be simpler, but hey! it works.
        for &constraint in &self.constraints {
            // skip constraints we're not involved in
            if constraint.0.iter().find(|&&i| i==chosen_var).is_none() {
                continue;
            }

            // for each variable involved in the constraint
            for var_idx in 0..3 {
                assigned_values[chosen_var] = Some(chosen_value);
                
                let var = constraint.0[var_idx];
                // that hasn't been assigned a value in a previous assignment
                if assigned_values[var].is_some() && var != chosen_var {
                    continue;
                }
                // try substituing all valid values, and see if constraints are violated
                valid_values[var].retain(|&value| {
                    // speculatively assign the value, to see if constraints have been violated
                    // (this will be undone before recursing)
                    // println!("{} <- {}", self.symbols[var], value);
                    assigned_values[var] = Some(value);

                    // Constraint Satisfied  IFF  a<b<c | b<c<a | c<a<b 
                    // Constraint Violated  IFF !( a<b<c | b<c<a | c<a<b )
                    // IFF !( (a<b & b<c ) | (b<c & c<a) | (c<a & a<b) )
                    // IFF !(a<b & b<c ) & !(b<c & c<a) & !(c<a & a<b)
                    // IFF (!a<b | !b<c ) & (!b<c | !c<a) & (!c<a | !a<b)
                    // IFF (a>=b | b>=c ) & (b>=c | c>=a) & (c>=a | a>=b)
                    // loop over the ANDs in the above loop
                    let violated = [
                        [0,1,2],
                        [1,2,0],
                        [2,0,1],
                    ].iter().all(|[x,y,z]|{
                        // a>=b | b>=c
                        // = iffy_cmp(a, b) | iffy_cmp(b, c)
                        let iffy_cmp = |&x, &y|{
                            match (assigned_values[constraint.0[x]], assigned_values[constraint.0[y]]) {
                                (Some(a), Some(b)) => a>=b,
                                (_, _) => false, // if either of them is None, they may receive a value or values in the future that satisfy the comparison
                            }
                        };
                        iffy_cmp(x, y) || iffy_cmp(y, z)
                    });

                    /*
                    print!("{} ?= {}", self.symbols[var], value);
                    print!(" -- -- ");
                    print!("({},{},{})", self.symbols[constraint.0[0]], self.symbols[constraint.0[1]], self.symbols[constraint.0[2]]);
                    print!(" -- -- ");
                    print!("({:?},{:?},{:?})", assigned_values[constraint.0[0]], assigned_values[constraint.0[1]], assigned_values[constraint.0[2]]);
                    print!(" -- -- ");
                    print!("vio: {} ", violated);
                    println!();
                    */

                    // valid_values will retain any values that do not violate this constraint.
                    !violated
                });

                // we're done speculating about assigned_values[var]
                assigned_values[var] = None;
            }
        }
        
        // we're done speculating about assigned_values[chosen_var]
        assigned_values[chosen_var] = None;
    }
    

    /// recursive implementation of solve
    fn solve_inner(&self, assigned_values: &mut [Option<usize>], old_valid_values: &Vec<Vec<usize>>) -> bool {
        // minimum remaining values heuristic (lower is better)
        let mrv = |var: usize| old_valid_values[var].len();
        // tie-breaker heuristic: most constraining variable (higher is better)
        let mcv = |var| self.constraints.iter().filter(|c| c.0.contains(&var)).count();
        // compare functions (both report better variables as lower)
        let mrv_cmp = |one, two| mrv(one).cmp(&mrv(two));
        // mcv() has higher is better, so we cmp backwards to make lower better.
        let mcv_cmp = |one, two| mcv(two).cmp(&mcv(one));

        // use minimum remaining values heuristic to choose a variable
        let chosen_var = match assigned_values.iter().enumerate()
            .filter_map(|(var, assn)| if assn.is_none() {Some(var)} else {None})
            // compare using heuristics
            .min_by(|&one, &two| mrv_cmp(one, two).then(mcv_cmp(one, two)))
        {
            Some(x) => x,
            None => return true, // We have no free variables, so the problem is solved!
        };

        // optional: followed by "Most Constraining Variable" tiebreaker (the variable with the largest number of involved constraints).
        // (choose one with lowest old_valid_values len)

        // "our" is referring to chosen_var
        // find the set of values that our chosen var can take on
        let our_values = &old_valid_values[chosen_var];
        // println!("{}'s Values: {:?}", chosen_var, our_values );

        // Calculate the result of assigning each value to this variable,
        //  so we can implement least constraining value quite easily below
        let mut futures = our_values.iter().filter_map(|&chosen_value| {
        /*let mut futures = iter::once(&match chosen_var{
            0 => 4, // A=4
            1 => 0, // B=0
            2 => 5, // C=5
            3 => 3, // D=3
            4 => 1, // E=1
            5 => 2, // F=2
            _ => unreachable!()
        }).map(|&chosen_value| {*/

            let mut valid_values = old_valid_values.clone();
            self.handle_constraints(chosen_var, chosen_value, &mut valid_values, assigned_values);

            // print!("{} := {};  ", self.symbols[chosen_var], chosen_value);
            // println!("{:?}", valid_values);
            if valid_values[chosen_var].is_empty() {
                // chosen_var := chosen_value violates a constraint
                None
            }else {
                Some((chosen_value, valid_values))
            }
        }).collect::<Vec<_>>();

        // mcv most constrained variable heuristic prioritizes the value you could assign to given
        //  variable x such that other variables still have the largest set of possible values.
        futures.sort_by_cached_key(|(_, valid_values)| valid_values.iter().map(Vec::len).sum::<usize>());

        for (chosen_value, valid_values) in futures {
            assigned_values[chosen_var] = Some(chosen_value);
            // let indent_depth = 4*assigned_values.iter().copied().filter(Option::is_some).count();

            // println!("{:>indent_depth$} := {}", self.symbols[chosen_var], chosen_value, indent_depth=indent_depth);
            if self.solve_inner(assigned_values, &valid_values) {
                return true;
            }
            // println!("{:>indent_depth$} != {} -- FAIL --", chosen_var, chosen_value, indent_depth=indent_depth);
        }

        // We failed to find a value, so we gotta unset ourselves to let the parent call explore
        assigned_values[chosen_var] = None;
        return false;
    }
    pub fn solve(&self) -> Option<Output> {
        // Initial state for recursive call
        let num_symbols = self.symbols.len();
        let mut assigned_values = vec![None; num_symbols];
        let valid_values: Vec<Vec<_>> = vec![(0..num_symbols).collect(); num_symbols];

        // solve_inner recursively solves the csat problem.
        if self.solve_inner(&mut assigned_values, &valid_values) {
            let symbols = self.symbols.clone();
            let values = assigned_values.into_iter().map(Option::unwrap).collect();
            Some(Output{symbols, values})
        } else {
            // no solution found :c
            None
        }
    }

    /// Verify that the solution provided solves this problem.
    pub fn is_solved_by(&self, solution: &Output) -> bool {
        // ensure solution is of the same order as our problem
        if solution.symbols.len() != self.symbols.len() {
            return false;
        }
        
        // apply values to variables in order they were specified in the input file.
        let num_symbols = self.symbols.len();
        let mut valid_values: Vec<Vec<_>> = vec![(0..num_symbols).collect(); num_symbols];
        let mut assigned_values = vec![None; num_symbols];
        for (var, &value) in solution.values.iter().enumerate() {
            // ensure each value assignment hasn't been prevented
            //  by a previous value assignment.
            if !valid_values[var].contains(&value) {
                return false;
            }

            self.handle_constraints(var, value, &mut valid_values, &mut assigned_values);
            assigned_values[var] = Some(value);
        }

        // ensure each value didn't violate constraint following it
        for (value, valid_set) in solution.values.iter().zip(valid_values.iter()) {
            if valid_set.get(0) != Some(value) {
                return false;
            }
        }

        true
    }
}

impl fmt::Display for Input {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // display symbols
        if let Some(symbol) = self.symbols.first() {
            write!(f, "{{ {}", symbol)?;
            for symbol in self.symbols.iter().skip(1) {
                write!(f, ", {}", symbol)?;
            }
            write!(f, " }}")?;
        } else {
            write!(f, "{{}}")?;
        }

        // seperate the symbols and constraints
        write!(f, "; ")?;

        // local helper function to format a constraint
        let fmt_constraint = |f: &mut fmt::Formatter<'_>, c: &Constraint| {
            write!(f,
                "({},{},{})",
                self.symbols[c.0[0]],
                self.symbols[c.0[1]],
                self.symbols[c.0[2]],
            )
        };

        // display constraints
        if let Some(constraint) = self.constraints.first() {
            write!(f, "{{ ")?;
            fmt_constraint(f, constraint)?;
            for constraint in self.constraints.iter().skip(1) {
                write!(f, ", ")?;
                fmt_constraint(f, constraint)?;
            }
            write!(f, " }}")
        } else {
            write!(f, "{{}}")
        }
    }
}

pub enum InputError {
    /// EOF was reached, but there was no partial input.
    NoMoreInputs,
    IoError(io::Error),
    CantParseSymbolCount(String, ParseIntError),
    MultipleDeclarationsOfSymbol(String),
    ExpectedMoreSymbols(usize, usize),
    ExpectedConstraintCount,
    CantParseConstraintCount(String, ParseIntError),
    ExpectedMoreConstraints(usize, usize),
    PartialConstraint(usize, usize),
    UndeclaredSymbol(String),
}

impl fmt::Debug for InputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use InputError::*;
        match self {
            NoMoreInputs => write!(f, "no more inputs; hit EOF"),
            IoError(e) => write!(f, "error reading: {}", e),
            CantParseSymbolCount(s, e) => write!(f, "can't parse {:?} as symbol count: {}", s, e),
            MultipleDeclarationsOfSymbol(s) => write!(f, "multiple declarations of symbol {:?}", s),
            ExpectedMoreSymbols(remaining, already_parsed) => write!(
                f, "expected {} more symbol(s); parsed {} before hitting EOF", remaining, already_parsed ),
            ExpectedConstraintCount => write!(f, "expected Constraint count; hit EOF"),
            CantParseConstraintCount(s, e) => write!(f, "can't parse {:?} as constraint count: {}", s, e),
            ExpectedMoreConstraints(remaining, already_parsed) => write!(
                f, "expected {} more constraint(s); parsed {} before hitting EOF", remaining, already_parsed ),
            PartialConstraint(symbol_count, already_parsed) => write!(
                f, "constraint #{} has only {} of 3 symbols",
                already_parsed+1,
                symbol_count
            ),
            UndeclaredSymbol(s) => write!(f, "symbol {:?} has not been declared", s),
        }
    }
}

impl From<io::Error> for InputError {
    fn from(e: io::Error) -> Self {
        InputError::IoError(e)
    }
}
