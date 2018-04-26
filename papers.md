# Scene Text Localization & Recognition Resources

## 1. Paper & Code

### 1.1 Text Recognition

##### 2018-CVPR

- Edit Probability for Scene Text Recognition
- **AON: Towards Arbitrarily-Oriented Text Recognition**

##### 2018-AAAI

- **Char-Net: A Character-Aware Neural Network for Distorted Scene Text Recognition**
- SEE- Towards Semi-Supervised End-to-End Scene Text Recognition
- SqueezedText- A Real-time Scene Text Recognition by Binary Convolutional Encoder-decoder Network

##### 2017-ICCV

- **Focusing Attention: Towards Accurate Text Recognition in Natural Images**

##### 2017-NIPS

- Gated Recurrent Convolution Neural Network for OCR

##### 2016-CVPR

- **Robust Scene Text Recognition With Automatic Rectification**
- Recursive Recurrent Nets with Attention Modeling for OCR in the Wild

##### 2016-AAAI

- Reading Scene Text in Deep Convolutional Sequences

##### 2016-NIPS

- Generative Shape Models: Joint Text Recognition and Segmentation with Very Little Training Data

##### 2015-CoRR

- **An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition** [`paper`](http://arxiv.org/pdf/1507.05717v1.pdf) [`code`](http://mclab.eic.hust.edu.cn/~xbai/CRNN/crnn_code.zip) [`github code`](https://github.com/bgshih/crnn)

##### 2015-ICLR

- Deep structured output learning for unconstrained text recognition 

##### 2014-NIPS

- Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition [`paper`](http://www.robots.ox.ac.uk/~vgg/publications/2014/Jaderberg14c/jaderberg14c.pdf)  [`homepage`](http://www.robots.ox.ac.uk/~vgg/publications/2014/Jaderberg14c/) [`model`](http://www.robots.ox.ac.uk/~vgg/research/text/model_release.tar.gz)

##### 2013-ICCV

- Photo OCR: Reading Text in Uncontrolled Conditions

### 1.2 Text Detection

##### 2018-CVPR

- Multi-Oriented Scene Text Detection via Corner Localization and Region Segmentation
- Geometry-Aware Scene Text Detection with Instance Transformation Network
- **Rotation-sensitive Regression for Oriented Scene Text Detection**
- Learning Markov Clustering Networks for Scene Text Detection

##### 2018-AAAI

- Feature Enhancement Network- A Refined Scene Text Detector
- PixelLink- Detecting Scene Text via Instance Segmentation

##### 2017-ICCV

- Deep Direct Regression for Multi-Oriented Scene Text Detection
- Self-organized Text Detection with Minimal Post-processing via Border Learning
- Single Shot Text Detector with Regional Attention
- WeText Scene Text Detection under Weak Supervision
- WordSup: Exploiting Word Annotations for Character based Text Detection

##### 2017-CVPR

- **EAST-An Efﬁcient and Accurate Scene Text Detector** [`code`](https://github.com/argman/EAST)
- Deep Matching Prior Network Toward Tighter Multi-oriented Text Detection
- **Detecting Oriented Text in Natural Images by Linking Segments**

##### 2017-AAAI

- TextBoxes: A Fast Text Detector with a Single Deep Neural Network [`paper`](https://arxiv.org/abs/1611.06779) [`code`](http://mclab.eic.hust.edu.cn/~xbai/CRNN/crnn_code.zip) [`github code`](https://github.com/bgshih/crnn)

##### 2016-CVPR

- A Text Detection System for Natural Scenes With Convolutional Feature Learning and Cascaded Classification
- Synthetic Data for Text Localisation in Natural Images [`paper`](http://www.robots.ox.ac.uk/~vgg/data/scenetext/gupta16.pdf) [`code`](https://github.com/ankush-me/SynthText) [`data`](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)
- Canny Text Detector: Fast and Robust Scene Text Localization Algorithm
- Multi-Oriented Text Detection With Fully Convolutional Networks

##### 2016-ECCV

- **CTPN: Detecting Text in Natural Image with Connectionist Text Proposal Network** [`paper`](https://arxiv.org/abs/1609.03605) [`code`](https://github.com/tianzhi0549/CTPN)

##### 2015-ICCV

- Text Flow: A Unified Text Detection System in Natural Scene Images
- FASText: Efficient Unconstrained Scene Text Detector [`paper`](https://pdfs.semanticscholar.org/2131/106318d4674bc9260e671c9f427bfc3f1029.pdf) [`code`](https://github.com/MichalBusta/FASText)

### 1.3 End-to-end Text Detection and Recognition

##### 2018-CVPR

- **FOTS: Fast Oriented Text Spotting with a Unified Network**
- **An End-to-End TextSpotter with Explicit Alignment and Attention** [`code`](https://github.com/tonghe90/textspotter)

##### 2017-ICCV

- Deep TextSpotter: An End-to-End Trainable Scene Text Localization and Recognition Framework
- **Towards End-to-end Text Spotting with Convolutional Recurrent Neural Networks**

##### 2017-AAAI

- Detection and Recognition of Text Embedding in Online Images via Neural Context Models [`paper`](https://www.google.com.hk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=0ahUKEwj0o9Cfqc3SAhVIXLwKHeSLABQQFggfMAE&url=https%3a%2f%2fraw%2egithubusercontent%2ecom%2fcmkang%2fCTSN%2fmaster%2faaai2017_cameraready%2epdf&usg=AFQjCNHcFvXZ0TDWnGnl20LD98V3LWxpbA) [`code`](https://github.com/cmkang/CTSN)

##### 2017-arxiv

- STN-OCR- A single Neural Network for Text Detection and Text Recognition

##### 2016-IJCV

- Reading Text in the Wild with Convolutional Neural Networks [`paper`](http://arxiv.org/abs/1412.1842) [`demo`](http://zeus.robots.ox.ac.uk/textsearch/#/search/)  [`homepage`](http://www.robots.ox.ac.uk/~vgg/research/text/)

##### 2014-TIP

- A Unified Framework for Multi-Oriented Text Detection and Recognition

##### 2014-ECCV

- Deep Features for Text Spotting [`paper`](http://www.robots.ox.ac.uk/~vgg/publications/2014/Jaderberg14/jaderberg14.pdf) [`code`](https://bitbucket.org/jaderberg/eccv2014_textspotting) [`model`](https://bitbucket.org/jaderberg/eccv2014_textspotting) [`GitXiv`](http://gitxiv.com/posts/uB4y7QdD5XquEJ69c/deep-features-for-text-spotting)

## 2. Datasets

#### [`COCO-Text (Computer Vision Group, Cornell)`](http://vision.cornell.edu/se3/coco-text/)   `2016`

63,686 images, 173,589 text instances, 3 fine-grained text attributes.

Task: text location and recognition

[`download`](https://github.com/andreasveit/coco-text)

#### [`Synthetic Word Dataset (Oxford, VGG)`](http://www.robots.ox.ac.uk/~vgg/data/text/)   `2014`

9 million images covering 90k English words

Task: text recognition, segmantation

[`download`](http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz)

#### [`IIIT 5K-Words`](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)   `2012`

5000 images from Scene Texts and born-digital (2k training and 3k testing images)

Each image is a cropped word image of scene text with case-insensitive labels

Task: text recognition

[`download`](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz)

#### [`StanfordSynth(Stanford, AI Group)`](http://cs.stanford.edu/people/twangcat/#research)   `2012`

Small single-character images of 62 characters (0-9, a-z, A-Z)

Task: text recognition

[`download`](http://cs.stanford.edu/people/twangcat/ICPR2012_code/syntheticData.tar)

#### [`MSRA Text Detection 500 Database (MSRA-TD500)`](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500))   `2012`

500 natural images(resolutions of the images vary from 1296x864 to 1920x1280)

Chinese, English or mixture of both

Task: text detection

#### [`Street View Text (SVT)`](http://tc11.cvc.uab.es/datasets/SVT_1)   `2010`

350 high resolution images (average size 1260 × 860) (100 images for training and 250 images for testing)

Only word level bounding boxes are provided with case-insensitive labels

Task: text location

#### [`KAIST Scene_Text Database`](http://www.iapr-tc11.org/mediawiki/index.php/KAIST_Scene_Text_Database)   `2010`

3000 images of indoor and outdoor scenes containing text

Korean, English (Number), and Mixed (Korean + English + Number)

Task: text location, segmantation and recognition

#### [`Chars74k`](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/)   `2009`

Over 74K images from natural images, as well as a set of synthetically generated characters

Small single-character images of 62 characters (0-9, a-z, A-Z)

Task: text recognition

#### `ICDAR Benchmark Datasets`

| Dataset                                                      | Description                                                  | Competition Paper                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [ICDAR 2015](http://rrc.cvc.uab.es/)                         | 1000 training images and 500 testing images                  | `paper`  [![link](https://www.lds.org/bc/content/shared/content/images/gospel-library/manual/10735/paper-icon_1150845_tmb.jpg)](http://rrc.cvc.uab.es/files/Robust-Reading-Competition-Karatzas.pdf) |
| [ICDAR 2013](http://dagdata.cvc.uab.es/icdar2013competition/) | 229 training images and 233 testing images                   | `paper`  [![link](https://www.lds.org/bc/content/shared/content/images/gospel-library/manual/10735/paper-icon_1150845_tmb.jpg)](http://dagdata.cvc.uab.es/icdar2013competition/files/icdar2013_competition_report.pdf) |
| [ICDAR 2011](http://robustreading.opendfki.de/trac/)         | 229 training images and 255 testing images                   | `paper`  [![link](https://www.lds.org/bc/content/shared/content/images/gospel-library/manual/10735/paper-icon_1150845_tmb.jpg)](http://www.iapr-tc11.org/archive/icdar2011/fileup/PDF/4520b491.pdf) |
| [ICDAR 2005](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2005_Robust_Reading_Competitions) | 1001 training images and 489 testing images                  | `paper`  [![link](https://www.lds.org/bc/content/shared/content/images/gospel-library/manual/10735/paper-icon_1150845_tmb.jpg)](http://www.academia.edu/download/30700479/10.1.1.96.4332.pdf) |
| [ICDAR 2003](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions) | 181 training images and 251 testing images(word level and character level) | `paper`  [![link](https://www.lds.org/bc/content/shared/content/images/gospel-library/manual/10735/paper-icon_1150845_tmb.jpg)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.332.3461&rep=rep1&type=pdf) |

## 3. Online OCR Service

| Name                                                         | Description      |
| ------------------------------------------------------------ | ---------------- |
| [Online OCR](https://www.onlineocr.net/)                     | API，free        |
| [Free OCR](http://www.free-ocr.com/)                         | API，free        |
| [New OCR](http://www.newocr.com/)                            | API，free        |
| [ABBYY FineReader Online](https://finereaderonline.com)      | No API，Not free |
| [Super Online Transfer Tools (Chinese)](http://www.wdku.net/) | API，free        |
| [Online Chinese Recognition](http://chongdata.com/ocr/)      | API，free        |

## 4. Blogs

- [Scene Text Detection with OpenCV 3](http://docs.opencv.org/3.0-beta/modules/text/doc/erfilter.html)
- [Handwritten numbers detection and recognition](https://medium.com/@o.kroeger/recognize-your-handwritten-numbers-3f007cbe46ff#.8hg7vl6mo)
- [Applying OCR Technology for Receipt Recognition](http://rnd.azoft.com/applying-ocr-technology-receipt-recognition/)
- [Convolutional Neural Networks for Object(Car License) Detection](http://rnd.azoft.com/convolutional-neural-networks-object-detection/)
- [Extracting text from an image using Ocropus](http://www.danvk.org/2015/01/09/extracting-text-from-an-image-using-ocropus.html)
- [Number plate recognition with Tensorflow](http://matthewearl.github.io/2016/05/06/cnn-anpr/) [`github`](https://github.com/matthewearl/deep-anpr)
- [Using deep learning to break a Captcha system](https://deepmlblog.wordpress.com/2016/01/03/how-to-break-a-captcha-system/) [`report`](http://web.stanford.edu/~jurafsky/burszstein_2010_captcha.pdf) [`github`](https://github.com/arunpatala/captcha)
- [Breaking reddit captcha with 96% accuracy](https://deepmlblog.wordpress.com/2016/01/05/breaking-reddit-captcha-with-96-accuracy/) [`github`](https://github.com/arunpatala/reddit.captcha)
- [文字检测与识别资源-1](http://blog.csdn.net/peaceinmind/article/details/51387367)
- [文字的检测与识别资源-2](http://blog.csdn.net/u010183397/article/details/56497303?locationNum=12&fps=1)
