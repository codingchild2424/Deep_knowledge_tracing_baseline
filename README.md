# Knowledge_tracing_baseline made by CoLA & EDM
This code is Deep knowledge Tracing(DKT) baseline code for Learning Analytics(LA) and Educational Data Mining(EDM) researchers.

And this code is made by CoLA & EDM(Community of Learning Analytics and Educational Data Mining), which is the research group of LA and EDM in South Korea.

If you want to contact with us, please send me a mail.

mail address: colaandedm@gmail.com

# Manual
## Python files

Python files are program to initialize the DKT training and predicting.

This program consist of 6 modules.

1. dkt.py: DKT model

2. trainer.py: class for training and validating

3. train.py: main file

4. utils.py: util class or functions

5. model.pth: trained model files

# Environments
We use Docker environments for develop my code.

The Docker Image is 'ufoym/deepo'.

'ufoym/deepo' has a lot of python package for deep learning.

# How to start?
1) Use terminal and type this code. You should check your path(use pwd).
```
$ python3 train.py --model_fn model.pth
```

2) If you want to use more options, then try this code and check the more options
```
$ python3 train.py --help
```

3) You can check your GPU can be available by using this code.
```
$ nvidia-smi
```

# References
1) Juno-hwang's Juno-dkt github page

https://github.com/juno-hwang/juno-dkt

2) Hyungcheol Noh's blog and github page

https://github.com/hcnoh/knowledge-tracing-collection-pytorch

https://hcnoh.github.io/2019-06-14-deep-knowledge-tracing

3) Kim, Ki Hyun's lectures in Fastcampus

https://fastcampus.co.kr/

4) Deep Knowledge Tracing Paper

Piech, C., Spencer, J., Huang, J., Ganguli, S., Sahami, M., Guibas, L., & Sohl-Dickstein, J. (2015). Deep knowledge tracing. arXiv preprint arXiv:1506.05908.