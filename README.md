1) Taking the relation “_synset_domain_usage_of” in WN18 as an example, there are total 597 head entities linked by it. The distance distribution among them in initial embedding space and position space is reported in the following table. 

      Distance | >0.6 | >0.8 | >1
      --------|--------|-------- | -------- 
      Initial Embedding Space| 99.9% | 99.5% | 95% 
      Position space| 39.6% | 19% | %7.5
      
   It is shown in above table that after the first-hop projection, a relation’s linked head entities are clustered in the position space as expected. 
  
   Finally, in the translation space, the distribution of its linked entities is as shown in following figure, which clearly shows that the entities are clustered into different groups and each one of the group corresponds to a specific semantic of the relation. 
   
   ![before](https://github.com/IJCAI-MSP/MSP/blob/master/images/before.png)

2) Q: Is MSP(H+D) a “TransH + TransD” KGE method?

   A: Generally, MSP is a joint KGE learning method that takes full advantages of existing KGE methods. So MSP(H+D) shouldn’t be simple regarded as a cascade “TransH + TransD” KGE method.
