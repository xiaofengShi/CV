# Computer Vision

## img 

## 1. Img to latex and charactor recognize

### 1.1 what you gets is what you  see

- The question is published here[Open AI-question source](https://openai.com/requests-for-research/#im2)

  - [Official resolution](http://lstm.seas.harvard.edu/latex/)
  - [Official repo-torch](https://github.com/harvardnlp/im2markup)

- Based on the paper[what you gets is what you  see](http://arxiv.org/pdf/1609.04938v1.pdf)

  - the model is ↓↓↓↓↓↓↓↓↓↓

    <img src="assets/network.png" width="400">

- This is a seq2seq model and use the trick of attention, gradient clip

  - The final output is a fully connected network and the loss of  model is softmax 

- The related program is here

  - [Img_to_latex](./img_latex_recognize/Img_to_latex)
    - This is the better repo referenced and modified 
  - [latex_to_img_test](./img_latex_recognize/latex_to_img_test)

- 

## 2. Text detection



