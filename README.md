

## Datasets

### Prompt Formatting

There are different ways to structure the prompt of a dataset. Next, are some
representative ways to do so:

#### Visual Q&A

1.  `<image> Question: <question> Answer:`
    
    Source: https://arxiv.org/pdf/2206.01718

2.  `<image> <question> Answer the question using a single word or phrase.`

    Source: https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md

3. `<image> Based on the above image, please answer the question. <question> Please provide an accurate answer within one word. The answer is: `

    Source: (Custom) https://arxiv.org/pdf/2401.10208

#### Image Captioning

1. `<image> Describe the image concisely.`

    Source: https://arxiv.org/pdf/2304.08485

2. `<image> Provide a one-sentence caption for the provided image.`

    Source: https://arxiv.org/pdf/2401.10208

#### Detailed Description

1. `<image> Describe the following image in detail.`

    Source: https://arxiv.org/pdf/2304.08485

2. `<image> Please describe the image in detail.`

    Source: https://arxiv.org/pdf/2401.10208

#### Complex Reasoning

1.  `<image> Question: <question> Answer:`
    
    Source: https://arxiv.org/pdf/2206.01718

2. `<image> <question>`

    Source: (No processing) https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md

#### Multiple-Choice Video Q&A

#### Video Description

#### Open-Ended Video Q&A

#### Multi-Image Q&A

#### Conversations

Source: (No processing) https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md

#### Audio Understanding

1. `<audio> Please describe this audio.`

    Source: https://arxiv.org/pdf/2307.08581 (Fig. 8 - 13)
