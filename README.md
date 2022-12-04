# NL2STL

This project is to transform human natural languages into Signal temporal logics (STL). Here to enhance the generalizability, in each natural language the specific atomic proposition (AP) is represented as Prop_1, Prop_2, etc. In this way, the trained model can be easier to transfer into various specific domains. One inference example is as the following:

Input natural language:

```
If ( prop_2 ) happens and continues to happen until at some point during the 176 to 415 time units that ( prop_1 ) , and also if ( prop_3 ) , then the scenario is equivalent to ( prop_4 ) .
```

Output Signal temporal logic:

```
( ( ( prop_2 until [176,415] prop_1 ) and prop_3 ) equal prop_4 )
```

## Description

Signal Temporal Logic (STL) is a formal language for specifying properties of signals. It is used to specify properties of continuous-time signals, such as signals from sensors or control systems, in a way that is precise and easy to understand.

STL has a syntax that is similar to the temporal logic used in computer science, but it is specialized for continuous-time signals. It includes operators for describing the values of a signal, as well as operators for combining and modifying those descriptions.

For example, the STL formula F[0, 2] (x > 0.5) specifies the property that the signal x is greater than 0.5 for all time points between 0 and 2 seconds. This formula can be read as "the signal x is eventually greater than 0.5 for a period of at least 2 seconds".

STL can be used to verify that a signal satisfies a given property, or to synthesize a controller that ensures that a signal satisfies a given property. It is a powerful tool for reasoning about the behavior of continuous-time systems.

While STL is quite powerful, humans are more familiar with defining the tasks via natural languages. Here we try to bridge this gap via fine-tuning large languages models.

## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
