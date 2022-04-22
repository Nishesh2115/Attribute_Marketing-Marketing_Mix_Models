# Attribute_Marketing-Marketing_Mix_Models

# Blog Link- https://medium.com/@nishesh.kumar/how-machine-learning-is-changing-the-digital-marketing-industry-attribute-marketing-models-e8a0d644e35d
# Business Problem

All the companies invest a lot of money on Digital Marketing these days on different channels like Email ADs, Google ADs, Instagram Feed, Youtube Ads, Facebook Feed etc.

Let's say Amazon as a company has invested 100 dollars on marketing on different channels like Google Ads(20 dollar ) Instagram feed(20 dollar), Facebook Feed(20 dollar) and on Youtube Ads(30 dollar) ,Now Amazon as a company wants to know,

"HOW MUCH CONTRIBUTION EACH CHANNEL DID TO THE REVENUE OF THE COMPANY?"

Now this question is very simple but the finding the solution to this could be very challenging because we need to understand that finally hit a customer to go for the product, what exactly works in the marketing campaign and more importantly why did not work out so that they can plan their Marketing Funds accordingly.

For example lets say from 80 percent of the sales from this marketing campaign comes from Youtube ADs, 10 percent comes from Google Search, 5 percent comes from Instagram, 5 percent from the Facebook.

Now if a company can get this detailed data about the percentage revenue from different channels then it will give more weightage of marketing budget to Youtube than other marketing channels.

This can literally help companies to understand the market in depth and will increase their revenue over time.

# Approach-1

# Rule Based/Intutive system

These are some simple rule based systems which are more like gut-feeling driven.
Here we can say clicks are more Important than Impressions, Impressions are simply when you see an ad but you did not click it.
Search ads are more influential than Display ads, this is a simple intution that if a person searches something on the internet, then there are more chances that he/she will buy it. But when a person just sees a ad on any social media there are less chances for that conversion.
One simple rule is LAST INTERACTION ATTRIBUTION MODEL, for example, lets say i went to facebook, i saw an AD of Headphone, then i went to youtube, i saw the same ad , then i went to Google and i searched those Headphones and i clicked on the Google Ad then i finally purchased the Headphone, so according to LAST INTERACTION ATTRIBUTION MODEL,We will give 90% credit to the last channel(Google ADs) and rest small percentages to rest of the channels.
Another simple rule based system could be FIRST ATTRIBUTION CHANNEL, this means we are assuming that the first ad which user saw was responsible for the conversion, here 90% credit is given to the first channel and rest to other channels, This normally happens with luxurious items or Any New Product, lets say A NEW MACBOOK where the channel could be the youtube presentation by APPLE.
Another could be a time decay model where we assume that with time the percentage credit to channels should Increase, for example lets say we showed the Smartphone ad to the user on Google Search AD, then we showed the ad to user on Instagram feed, then we showed the ad to the user on Facebook feed then we showed the ad on youtube. Now lets say user bought the smartphone, now according to the Time Decay model we will give maximum credits percentage of conversion to Youtube lets say 60%, then Facebook lets say 20%, then Instagram lets say 10%, then at last Google Ads lets say 5%.
# Approach-2

# Regression Based Machine Learning Models

Why not map this problem as a Machine Learning Problem???

So lets say we have some old data which tells us about the revenue by different channels and the total sales for that campaign, now we can build a simple regression model where features will be the different channels and sales will be the target variable.

Regression will tell us how independent features(different channels) are mathematically related to dependent features (sales).

y= x1(Youtube) + x2(Facebook) + x3(Instagram) + x4(GoogleAds)

If I just find the values x1,x2,x3 and x4 then it will give me a good approximation of, how much Youtube is contibuting, how much Facebook is contributing, how much Instagram is contibuting and how much GoogleAds is contibuting.

Thats what we want,Right...

Now before going to another approach which is a Marcov Chains, lets see some code of regression approach for Attribute Modelling

# Data for Regression Based Attribute Model

I have downloaded the data from kaggale, this seems a dummy data but not bad for understanding the regression approach.

Link for the data:- https://www.kaggle.com/datasets/sazid28/advertising.csv

Data has 3 features or channels , First is TV, Second is Radio, Third is Newspaper. In this data we have revenue amount by different channels and then we have a total sales column.

For example- In first row, Total Sales is 22.1 lakhs, 2 lakh 30 thousand comes from TV, 37 thousand comes from Radio, 69 thousand comes from Newspaper, other amount can be assumed from direct sales.

# Performance Metric

We have to use the Performance Metric for Regression.

RMSE(ROOT MEAN SQUARE ERROR)
R^2 (Coefficient of Determinant)
Adjusted R^2

# Approach-3 (Markov-Chain Model)

Our aim is very simple, we want to understand that what marketing channel contribute to sales and in what proportion.

We will see, how MARKOV CHAIN MODEL can help us to solve this problem.

Before going to Markov Chain Model, lets understand the data for this case study.

# Data

You can download the data from here- https://drive.google.com/file/d/1KhT9fp9CCAhc6j7y6W6BeMNZvsXM_9NR/view?usp=sharing

Here we have a interesting data, we have a data in which a ad have been showed to people through different channels like instagram,online display, paid search, youtube video, facebook.

Lets discuss Features...

Feature-1--> Cookie

To identify every user, we have a cookie and respective channel on which the ad has been shown.

Feature-2--> Time

This is just the time when a particular ad was shown to a user.

Feature-3--> Interaction

For every ad shown to every user, We have a interaction features which tells us whether that ad was just a impression or that ad converted to a impression.

Feature-4--> Conversion

This feature tells us whther there was a coversion or not.

Feature-5--> Channel

To single user, we are showing ads to multiple channels through a path,channel could be Instagram, Facebook or Youtube Video

For example,

There could be a chance that a user has seen a ad on instagram first, then on facebook and then on youtube and finally he purchased that item, so understanding that path and understanding what channel causes that conversion is very important.

# Markov Chain Approach

Let's formulate our problem in a way that Marcov chains can help us to solve the problem

Here we have some states, lets discuss them one by one...

S0- It is the starting state, where we are assuming customer had not seen an ad yet

SF- It is a state where we have shown the ad to the user through FACEBOOK Ads

SF- It is a state where we have shown the ad to the user through GOOGLE Ads

SNP-It is a state where customer has not purchased an item, so this is NO-CONVERSION State.

SP- It is a state where customer has purchased an item, so this is a CONVERSION State.

</b>

From the above diagram, we can see that without showing any ad, there is 40% sale value, this is called the direct sale, so we can say that,

P(SP/S0)=0.4

(It means the probability that customer will directly go to purchase state given that he was at 0th state or starting state is 0.4)

These weights are conditional probabilities.

P(SNP/S0)=0.1

(It means the probability that customer will directly will not purchase is 0.1)

P(SF/S0)=0.1

(It means there is a 10% probability that we show a FB ad to a user)

P(SP/SF)=0.1

(It means the probability of a customer that he will purchase given that he has seen the FB ad is 0.1)

# Now the question is, How do we fill these probabilities??

We do it from data, lets say we have 10k customers at S0 state, now we decide to show some of them FB ads, some of them Google ads, then see how many of them converted and how many did not get converted, we will fil these probabilities emperically/through observation.

These are simple conditional probabilities, by observation we found that 4k people end up purchasing without showing any ad, it means what is the probability that customer will go to purchase state given that he/she is on initial stage, it is simply 4k/10k which is 0.4.

Also there is a temporal angle to it that if customer comes to website and does not purchase in 7 days, I will say customer has not purchased.

# MemoryLess Property of Markov Chains

Now markov chain has this MemoryLess property, lets discuss that in detail...

P(Sj/Sj-1,Sj-2.....S0)=P(Sj/Sj-1)

Here this equation simply says that, Probabilty of going to STATE Sj only depends on the previous state,it does not care about history of other states, as it is shown above in the formula, Probability of going Sj will only depend on the Sj-1 and not on earlier states or path.

Now the question is, DOES MemoryLess Property actually hold in our marketing problem?

Memeoryless property states that probability of reaching to the current state just depends upon the previous state and not the path previous to that, for example

Lets say a user was shown a Facebood ad first, then a Google ad, Now we want to know the probabilty that he will purchase the product, so according to markov memory less property, probability of coversion depends upon the previous state which is a Google ad in this case so HERE IN THIS CASE

P(CONVERSION/GOOGLE_AD, FACEBOOK_AD)=P(CONVERSION/GOOGLE_AD)=0.3(given in the diagram)

Marcov memory less property/assumption makes our life very simple as it neglects the path but if we think logically probability of conversion here just do not depends on Google_ad, it also depends upon Facebook_ad but here we are neglecting that.

Now when we have this property, we can come with the overall P(CONVERSION), which is

P(SP)=(0.4)+(0.10.1)+(0.10.10.3)+(0.40.3)=0.533

SP here is simply the purchase state.

Again this multiplication we can only do when we are assuming that probability of reaching SP only depends on the previous state.

Now coming to Attribution Modelling Problem

We have started with the very basic problem, we wanted to understand the contribution of each and every marketing channel in the sales so that we can improve the system.

We wanted to understand the impact of a particular channel, lets say i want to understand the how important Google ads are in above example, Lets Just remove Google from the markov diagram and then see how will it change the conversion probabilty.

P(SP)=(0.4)+(0.1*0.1)=0.41

P(SP) WHEN GOOGLE ADS WAS THERE =0.533

P(SP) WHEN NO GOOGLE ADS= 0.41

EFFECT OF REMOVAL= 1-(0.41/0.533)=0.23

Lets say earlier i have 1000 conversions, now after removing google conversion will be 1000*(0.41/0.533)=773.5, Now i will have 773 conversion.

23.07% Percentage of sales would be lost if we do not use Google Ads


# Drawbacks of Marcov Chains

The major issue with markov chains is its Memoryless Assumption, if we do not use this assumption them time complexity would be exponential.

# How will we compute Edge Weight/Probabilities?

Lets understand this by a simple example, imagine a company wants to spend $100k dollars on a marketing campaign, tall the money spent on different channels for next 30 days, after the campaign ended in 30 days , company waited for more 7 days for conversion, In this whole time, few people will go for direct sales, few will go after Google Ads, Few will go after Facebok Ad and so on.

Based on this data, we will emperically add probabilies and find the conditional probabilites or edge weight.

