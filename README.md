
<h1> Faces of Non-Existent Strangers - Deep Convolutional Generative Adversarial Networks (D.C. G.A.N.S.) </h1>

> In recent years, the potential of Neural Networks has extended into impressive territory, allowing for unprecedented function. With his famous paper published in 2014, Ian Goodfellow introduced a new stride for Neural Networks, termed G.A.N.s, or Generative Adversarial Networks, which ushered in a new methodology of developing algorithms through *mimicking human to human interaction*. 

> Using a dataset taken from the Chinese University of Hong Kong, we develop a **Deep Convolutional Generative Adversarial Network** to generate new images of human faces previously *non-existent* by operating as two individual neural networks trying to "outsmart" each other - a discriminator and a generator. Details as to how a G.A.N.s operates are discussed below. 
  
<h2> How It Works </h2> 

> G.A.N.s are defined by their characteristic neural networks - **a discriminator and a generator** - both of which operate in a relationship similar to a police officer and an art forger. In this scenario, the discriminator takes the role of the officer, while the generator takes the role of an art forger. *Over multiple iterations, the two neural networks play a game of cat and mouse* , in which the art forger (generator), after learning on our data, creates a fake piece of art, which is then mixed with other pieces of art (our data) and given to the officer (discriminator) who studies both, unknowing of the real and fake pieces, and determines which one's are real, and which are fake. 

> From there, the results are fed back to the art forger, who studies where he went wrong, and tries again. This process repeats tens, hundreds, possibly thousands of times until we attain results in which the *artificially generated photos are indiscernable to the real data* fed to the discriminator, and consequently, to us. With that, let's take a dive into our work.
  
<h2> The Goal </h2> 
  
> Celebrities are often the talk of society. Their voices are projected through every website and advertisement, and their photos plastered across magazines and billboards. And, rather unfortunately, it is that second reason which makes them useful within this project. With so many abundant photos, we are able to have a datascientists dream - an abundant, clean, large dataset. 

> Using data downloaded from The Chinese University of Hong Kong found [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), we are able to take over *204,000 images* of celebrity faces and teach our neural networks how to create accurate, natural faces. 
  
<h2> Required Imports </h2>
These include:

- Tensorflow
  >An open-source artificial intelligence library, particularly used for training deep neural networks

- Keras
  >An open-source library that provides an interfaces for our tensorflow based neural networks

- Numpy
  >A library to support large arrays and functions, which will be required in analyzing our dataset

- MatplotLib
  >A plotting library to visualize certain features


<h2> Data Analysis and Exploration </h2>
Loading in the dataset from Google Drive, and using Matplotlib, we note:

  - Orientation of faces in each image (headshot, body shot, etc.)
  - The size and shape of each image
  - Diversity of our dataset
    - This is particularly important given the context of our project, in which we want to generate images of all people, not a specific group, ethnicity, race or gender
  - Any facial coverings on majority of photos

A peak into our dataset
<img width="1245" alt="Screen Shot 2021-10-15 at 2 16 04 PM" src="https://user-images.githubusercontent.com/69823896/152109012-ecbff1cd-2ef8-46a8-a5e7-24d9fbacc144.png">

Since our data is exquisitely prepared and cleaning, we can immediately build our model


<h2> Building Our G.A.N.s Model </h2>

**Building our Discriminator** 

We begin with the input shape of our image, measured in height and width of pixels, as a parameter of the function, which is of size 128 by 128. This is equal across all photos in the dataset. We create our discriminator following a sequential neural netwowrk model, essentially layering each level of our model on top of each other. 
  - With every line of code, our data is essentially passing from the previous line, undergoing the specified transformation present within the layer of our current line of code, and then passing to the next line of code with it's specified layer
- Following this process, we downsize our sampled image from 128 by 128 pixel dimensions to an image of 8 by 8 pixels using LeakyReLU, a common activation function for G.A.N.s applications. 
- We then flatten our 8 by 8 image for our classifier, so that the information is in a 1 dimensional array for our neural networks to interpret. 
- Lastly we implement an Nadam optimizer, a variation on the standard Adam optimizer with a specialized gradient decent function to improve convergence. 

<img width="898" alt="Screen Shot 2021-10-15 at 2 16 23 PM" src="https://user-images.githubusercontent.com/69823896/152110880-bcaac25c-5595-44ed-89e5-3772c74b962b.png">

**Building our Generator** 

Now, we can start defining our generator. Essentially, what we will do here is the reverse of our discriminator, using the first layers to transpose our discriminator output. 
> By the 4th layer, we are back to our original 128 by 128 dimension shape. 
Once we have our generator model, we then define our generator's input, including a function to generate our fake samples to trick our discriminator.

<img width="690" alt="Screen Shot 2021-10-15 at 2 16 53 PM" src="https://user-images.githubusercontent.com/69823896/152113841-5fb4615a-1df0-4daa-901f-c469a1f00c94.png">

**Building our G.A.N.**

This portion of our model is where our adversarial network recipe comes together. Creating a function for our G.A.N. for ease of use, we set the weights of our model to be untrainable such that the delicate relationship between our discriminator and generator is unblemished. 
We as well implement another sequential model that *connects our discriminator and generator together as layers* in our G.A.N.
- We then set specific parameters, such as learning rate and decay rate
- We as well create a function to *retrieve* real samples and settle them into a dataset, into a class of either 1 (True) or 0 (False)

**Visualizing our Work**

Following suit is our function for generating and plotting our images in a 10 by 10 plot, and our performance evaluator. This takes in our real samples, evaluates our discriminator on the real examples, then takes in fake examples and feeds them into our discriminator, and summaries our performance.


<img width="636" alt="Screen Shot 2021-10-15 at 2 17 30 PM" src="https://user-images.githubusercontent.com/69823896/152114924-96498dbb-9713-4d9f-b33f-4b4070323eb0.png">


<h2> Training </h2>

Finally, we can train our G.A.N.s and get some results. Here we we iterate through the length of our epochs and batches across our training set, randomly selecting samples, updating our discriminator weights, generating fake samples, and updating our discriminator weights. 

We then prepare our input for the generator, creating inverted labels for the fake samples so they appear real, and update our generator through based on our discriminators error rate. Towards the end of the function, we summarize our loss and evaluate how well our gans performed on this iteration.


<img width="985" alt="Screen Shot 2021-10-15 at 2 18 02 PM" src="https://user-images.githubusercontent.com/69823896/152115476-f79bf8be-76d1-4eb7-b760-1f847d22db6a.png">

<h2> Results </h2> 

Though seemingly unfavorable, the results are rather telling. Our G.A.N. model runs, and runs properly, improving with each epoch. However, one unforseen circumstance is the computational expense and toll running such a model requires. Despite a long run-time, the results are a reflection of equipment limitations. Yet, as we can see in the short, ten iterations, our model *worked*, and continuously *improved* with each iteration, demonstrating a promising model despite the visual results 

<img width="622" alt="Screen Shot 2021-10-15 at 1 44 29 PM" src="https://user-images.githubusercontent.com/69823896/152115617-ab670bc3-8b53-4dd9-8f88-821617043173.png">


<h2> Conclusion </h2>

In conclusion, our model performed exceedingly well, despite the limitations. Given less constraints, I am certain that the pattern of improvement seen across 10 iterations could replicate at the same rate across hundreds of epochs, resulting in greatly improved, visually identical photos of faces.  

<h2> Resources </h2>

Dataset, found [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). 

Jupyter Notebook, found [here](https://colab.research.google.com/drive/13shMGApMXWm6cGMyBbKRxUSdYyRHNWTQ?usp=sharing)

Youtube Videos, found
  - [here](https://youtu.be/Sw9r8CL98N0)
  - [here](https://youtu.be/Z6rxFNMGdn0)

