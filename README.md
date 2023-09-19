# NL2TL
Webpage: https://yongchao98.github.io/MIT-realm-NL2TL/

Demo Website: http://realm-02.mit.edu:8444

Paper Link: https://arxiv.org/pdf/2305.07766.pdf

Dataset Link: https://drive.google.com/drive/folders/10F-qyOhpqEi83o9ZojymqRPUtwzSOcfq?usp=sharing

Model Link: [https://drive.google.com/drive/folders/1vSaKOunMPA3uiOdx6IDbe-gmfREXQ9uO?usp=share_link](https://drive.google.com/drive/folders/1ZfZoYovWoy5z247VXZWZBniNrCOONX4N?usp=share_link)

To access the Demo Website, please send email to ycchen98@mit.edu or yongchaochen@fas.harvard.edu for **password**

This project is to transform human natural languages into Signal temporal logics (STL). Here to enhance the generalizability, in each natural language the specific atomic proposition (AP) is represented as prop_1, prop_2, etc. In this way, the trained model can be easier to transfer into various specific domains. The APs refer to some specific specifications like grad the apple, or go to the room.

Also in the current work, the co-reference is not considered. Therefore, **each prop_i should only appear once in each sentence**. One inference example is as the following:

Input natural language:

```
If ( prop_2 ) happens and continues to happen until at some point during the 176 to 415 time units that ( prop_1 ) , and also if ( prop_3 ) , then the scenario is equivalent to ( prop_4 ) .
```

Output Signal temporal logic:

```
( ( ( prop_2 until [176,415] prop_1 ) and prop_3 ) equal prop_4 )
```

The operations we used are U(until), F(finally), G(globally), |(or), &(and), ->(imply), <->(equal), negation. Also we allow the time interval definition, like U[0,5], F[12,100], and G[30,150]. The time numer right now is constrained into integer, and can use infinite to express all the time in the future, like [5,infinite]. The following are the illustrations. More NL-TL pair examples at https://drive.google.com/file/d/1f-wQ8AKInlTpXTYKwICRC0eZ-JKjAefh/view?usp=sharing
```
prop_1 U[0,5] prop_2 : There exits one time point t between 0 and 5 timesteps from now, that prop_1 continues to happen until at this timestep, and prop_2 happens at this timestep.
```
```
F[12,100] prop_2 : There exits one time point t between 12 and 100 timesteps from now, that prop_2 happens at this timestep.
```
```
G[30,150] prop_2 : For all the time between 30 and 150 timesteps from now, that prop_2 always happens.
```
```
prop_1 -> prop_2 : If prop_1 happens, then prop_2 also happens.
```
```
prop_1 <-> prop_2: prop_1 happens if and only if prop_2 happens.
```

## Description

Signal Temporal Logic (STL) is a formal language for specifying properties of signals. It is used to specify properties of continuous-time signals, such as signals from sensors or control systems, in a way that is precise and easy to understand.

STL has a syntax that is similar to the temporal logic used in computer science, but it is specialized for continuous-time signals. It includes operators for describing the values of a signal, as well as operators for combining and modifying those descriptions.

For example, the STL formula F[0, 2] (x > 0.5) specifies the property that the signal x is greater than 0.5 for all time points between 0 and 2 seconds. This formula can be read as "the signal x is eventually greater than 0.5 for a period of at least 2 seconds".

STL can be used to verify that a signal satisfies a given property, or to synthesize a controller that ensures that a signal satisfies a given property. It is a powerful tool for reasoning about the behavior of continuous-time systems.

While STL is quite powerful, humans are more familiar with defining the tasks via natural languages. Here we try to bridge this gap via fine-tuning large languages models.

## Getting Started

### Dependencies

* The inference model should run on GPU, you can run the notebook file Run.ipynb on Google Colab, or run_trained_model.py on your own GPU environment.
* As for setting the environment, here we install our environmrnt via Minoconda. You can first set up it via https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
* Then it is time to install packages:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pip
conda install python
conda install numpy
conda install pandas
pip install transformers
pip install SentencePiece
```

### Installing

* First download the whole directory with command
```
git clone git@github.com:yongchao98/NL2TL.git
```
* Then download the trained wieghts (e.g., checkpoint-62500) of our model in [https://drive.google.com/file/d/19uiB_2XnnnVmDInaLbQeoZq25ghUdg4D/view](https://drive.google.com/drive/folders/1ZfZoYovWoy5z247VXZWZBniNrCOONX4N?usp=sharing)
* After downloading both the code and model weights, put the model weights checkpoint-62500 into your self-defined directory.

### Other codes and datasets

* As for other codes and datasets published on github, please read the **Illustration of Code and Dataset.pdf** for specific explanation of their utilities.

## Authors

Contributors names and contact info

Yongchao Chen (Harvard University, Massachusetts Institute of Technology, Laboratory of Information and Decision Systems)

yongchaochen@fas.harvard.edu or ycchen98@mit.edu

## Citation for BibTeX

@article{chen2023nl2tl,
  title={NL2TL: Transforming Natural Languages to Temporal Logics using Large Language Models},
  author={Chen, Yongchao and Gandhi, Rujul and Zhang, Yang and Fan, Chuchu},
  journal={arXiv preprint arXiv:2305.07766},
  year={2023}
}
}

## Version History

* 0.1
    * Initial Release on May 12, 2023

## License

This corresponding paper of this project will be attached here in the next months. This project can only be commercially used under our permission.

## Recommended Work

[AutoTAMP: Autoregressive Task and Motion Planning with LLMs as Translators and Checkers](https://arxiv.org/pdf/2306.06531.pdf)

[Scalable Multi-Robot Collaboration with Large Language Models: Centralized or Decentralized Systems?](https://yongchao98.github.io/MIT-REALM-Multi-Robot/)

