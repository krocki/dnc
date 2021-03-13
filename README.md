# Differentiable Neural Computer in Numpy
Implementation of Differentiable Neural Computer https://www.nature.com/articles/nature20101 as close as possible to the description in the paper. Task: char-level prediction. The repo also includes simple RNN (rnn-numpy.py) and LSTM (lstm-numpy.py). Some external data (ptb, wiki) needs to be downloaded separately.

### quick start
```
OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=8   python dnc-debug.py
```

These versions are cleaned up.
```
python rnn-numpy.py
python lstm-numpy.py
python dnc-numpy.py
```

#### Credits
RNN code based on original work by A.Karpathy (min-char-rnn.py)

gist:
https://gist.github.com/karpathy/d4dee566867f8291f086

post: 
http://karpathy.github.io/2015/05/21/rnn-effectiveness/

### Features
* RNN version still depends only on numpy
* Added batching
* Modified RNN into an LSTM
* Includes gradient check

### DNC
###### Implemented
* LSTM-controller
* 2D memory array
* content-addressable read/write
###### Issues
* softmax on key similarity causes crashes (divide by 0) - if you experience this, need to restart
###### TODO
* dynamic memory allocation/free
* faster implementation (PyTorch?)
* saving the model
* sample

### Sample output:

Time, iteration, BPC (prediction error -> bits per character, lower is better), processing speed
```
  0: 4163.009 s, iter 104800, 1.2808 BPC, 1488.38 char/s
```

### sample from the model (alice29.txt):
```
 e garden as she very dunced.
                  
  Alice fighting be it.  The breats?
              here on likegs voice withoup.
                                                                               
  `You minced more hal disheze, and I done hippertyou-sage, who say it's a look down whales that
his meckling moruste!' said Alice's can younderen, in they puzzled to them!'
     
  `Of betinkling reple bade to, punthery pormoved the piose himble, of to he see foudhed
just rounds, seef wance side pigs, it addeal sumprked.
                                                                                    
  `As or the Gryphon,' Alice said,
Fith didn't begun, and she garden as in a who tew.'
    
  Hat hed think after as marman as much the pirly
startares to dreaps
was one poon it                                                                           
out him were brived they                                                        
proce?                                                                                    
                                                                                 
                                                                                          
  CHAT, I fary,' said the Hat,' said the Divery tionly to himpos.'               
                                                                                          
  `Com, planere?"'                                                               
                                                                                          
  `Ica--'                                                                        
            Onlice IN's tread!  Wonderieving again, `but her rist,' said Alice.           
                                                                                 
                                                                                          
  She                                                                            
sea do voice.                                                                             
                                                                                 
  `I'mm the Panthing alece of the when beaning must anquerrouted not reclow, sobs to      
                                                                                 
  `In of queer behind her houn't seemed                                                   

```

### numerical gradients to check backward pass (rightmost column should have values < 1e-4);
the middle column has ranges of computed analytical and numerical gradients (these should match more/less)

```
----
GRAD CHECK

Wxh:            n = [-1.828500e-02, 5.292866e-03]       min 3.005175e-09, max 3.505012e-07
                a = [-1.828500e-02, 5.292865e-03]       mean 5.158434e-08 # 10/4
Whh:            n = [-3.614049e-01, 6.580141e-01]       min 1.549311e-10, max 4.349188e-08
                a = [-3.614049e-01, 6.580141e-01]       mean 9.340821e-09 # 10/10
Why:            n = [-9.868277e-02, 7.518284e-02]       min 2.378911e-09, max 1.901067e-05
                a = [-9.868276e-02, 7.518284e-02]       mean 1.978080e-06 # 10/10
Whr:            n = [-3.652128e-02, 1.372321e-01]       min 5.520914e-09, max 6.750276e-07
                a = [-3.652128e-02, 1.372321e-01]       mean 1.299713e-07 # 10/10
Whv:            n = [-1.065475e+00, 4.634808e-01]       min 6.701966e-11, max 1.462031e-08
                a = [-1.065475e+00, 4.634808e-01]       mean 4.161271e-09 # 10/10
Whw:            n = [-1.677826e-01, 1.803906e-01]       min 5.559963e-10, max 1.096433e-07
                a = [-1.677826e-01, 1.803906e-01]       mean 2.434751e-08 # 10/10
Whe:            n = [-2.791997e-02, 1.487244e-02]       min 3.806438e-08, max 8.633199e-06
                a = [-2.791997e-02, 1.487244e-02]       mean 1.085696e-06 # 10/10
Wrh:            n = [-7.319636e-02, 9.466716e-02]       min 4.183225e-09, max 1.369062e-07
                a = [-7.319636e-02, 9.466716e-02]       mean 3.677372e-08 # 10/10
Wry:            n = [-1.191088e-01, 5.271329e-01]       min 1.168224e-09, max 1.568242e-04
                a = [-1.191088e-01, 5.271329e-01]       mean 2.827306e-05 # 10/10
bh:             n = [-1.363950e+00, 9.144058e-01]       min 2.473756e-10, max 5.217119e-08
                a = [-1.363950e+00, 9.144058e-01]       mean 7.066159e-09 # 10/10
by:             n = [-5.594528e-02, 5.814085e-01]       min 1.604237e-09, max 1.017124e-05
                a = [-5.594528e-02, 5.814085e-01]       mean 1.026833e-06 # 10/10
```

# Links
https://github.com/JoergFranke/ADNC
https://github.com/RobertCsordas/dnc
https://github.com/SeemonJ/reinvent-dnc
https://github.com/bhpfelix/DNC-NumPy
https://github.com/llealgt/DNC
https://github.com/yashbonde/Differentiable-Neural-Computer
https://github.com/vpegasus/dnc
https://github.com/SiliconSloth/ADNC
https://github.com/ivannz/krocki_dnc
https://github.com/pchlenski/dnc-regex
https://github.com/ksluck/sl-dnhc
https://github.com/jaspock/dnc
https://github.com/brandontrabucco/program-hdc
https://arxiv.org/pdf/1505.00521.pdf
file:///home/me/Downloads/ntm_rl.pdf


