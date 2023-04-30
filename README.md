# NL2TL
Webpage: https://yongchao98.github.io/MIT-realm-NL2TL/

Demo Website: http://realm-02.mit.edu:8444

To access the Demo Website, please send email to ycchen98@mit.edu or yongchaochen@fas.harvard.edu

This project is to transform human natural languages into Signal temporal logics (STL). Here to enhance the generalizability, in each natural language the specific atomic proposition (AP) is represented as prop_1, prop_2, etc. In this way, the trained model can be easier to transfer into various specific domains. The APs refer to some specific specifications like grad the apple, or go to the room.

Also in the current work, the co-reference is not considered. Therefore, each prop_i should only appear once in each sentence. One inference example is as the following:

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
* Then download the trained wieghts (e.g., checkpoint-72000) of our model in [https://drive.google.com/file/d/19uiB_2XnnnVmDInaLbQeoZq25ghUdg4D/view](https://drive.google.com/drive/folders/1ZfZoYovWoy5z247VXZWZBniNrCOONX4N?usp=sharing)
* After downloading both the code and model weights, put the model weights model_state.pt into the directory eng2ltl_weights_11_28_word_infix

### Executing program

* We can run the code via both Run.ipynb and run_trained_model.py. Their functions are the same.
* To run the notebook Run.ipynb, first move the whole NL2STL directory into the Google Drive, and then open the Run.ipynb and connect to GPU. Connect the notebook to your Google drive by clicking Mount Drive in the left panel. Then in the first cell of Run.ipynb, change the path to your NL2STL directory in your Google drive, so that the following codes are run in the NL2STL dir. You are all set to run now.
* To run run_trained_model.py, you can directly use 
```
python run_trained_model.py
```
Or use the shell submission
```
sbatch run_fedcon_test.sh
```
Be careful that you need to use GPU.

## Setting your own input natural language

In one part of Run.ipynb or run_trained_model.py, we have set three methods to modify the test_sentence.
* The first is to directly change the sentence list
* The second is to read from the excel file. We have put the example file example_excel_nl.xlsx into the NL2STL
* The third is to read from txt file. We have put the example file example_excel_nl.txt into the NL2STL

## Authors

Contributors names and contact info

Yongchao Chen (Harvard University, Massachusetts Institute of Technology, Laboratory of Information and Decision Systems)

yongchaochen@fas.harvard.edu or ycchen98@mit.edu

## Version History

* 0.1
    * Initial Release on Dec 3

## License

This corresponding paper of this project will be attached here in the next months. This project can only be commercially used under our permission.

## Acknowledgments

