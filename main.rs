//! seasat constraint solver
//!
//! This file contains the entrypoint and commandline interface.
//! 
//! Magnus Larsen 2021

mod csat;

use std::fs::File;
use std::io::{BufRead, BufReader};
use csat::{Input, InputError};

/// Handles commandline interface and program lifecycle
fn main() {
    if std::env::args().count() > 2 {
        // print help when too many arguments are given
        eprintln!("seasat: too many commandline arguments");
        eprintln!("one argument: input filename (or '-' for stdin)");
        std::process::exit(1)
    }
    // Parse commandline arg
    let filename = std::env::args().skip(1).next().unwrap_or_else(||"A2-input.txt".to_owned());

    // Read file or stdin
    let mut file_maybe = None;
    let mut stdin_maybe = None;
    let mut stdin_lock_maybe = None;
    let mut input: &mut dyn BufRead  = if filename == "-" {
        // NOTE: Because the cubs and pups only have Rust 1.41,
        // I can't use Option::insert, but get_or_insert can be used.
        stdin_lock_maybe.get_or_insert(
            stdin_maybe.get_or_insert(std::io::stdin()).lock()
        )
    } else {
        file_maybe.get_or_insert(
            BufReader::new(
                File::open(&filename).unwrap_or_else(|e|{
                    eprintln!("seasat: could not open file '{}': {}", filename, e);
                    std::process::exit(1);
                })
            )
        )
    };

    // Parse inputs
    let (inputs, termination) = Input::many_from_read(&mut input);
    let num_inputs = inputs.len();

    // Solve inputs, and display the results to the user
    for input in inputs.into_iter() {
        println!("IN:  {}", input);
        match input.solve() {
            Some(solution) => {
                println!("OUT: {}", solution);
                assert!(input.is_solved_by(&solution));
            },
            _ => println!("COULD NOT SOLVE"),
        }
        println!();
    }

    // If there was an error parsing an input, report it.
    std::process::exit(match termination {
        InputError::NoMoreInputs if num_inputs == 0 => { eprintln!("seasat: file '{}' has no input problems!", filename); 1 },
        InputError::NoMoreInputs => 0,
        e => { eprintln!("seasat: Could not parse input #{} from file '{}': {:?}", num_inputs + 1, filename, e); 1 }
    });
}
