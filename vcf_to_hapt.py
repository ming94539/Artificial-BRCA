#!/usr/bin/env python
# coding: utf-8

# In[1]:


import allel


# In[25]:


# callset = allel.read_vcf('f5_chr17_brca1_copd_hmb.vcf')
callset = allel.read_vcf('f8_brca2_2504samples_805snps.vcf') 
#brca
#SN	0	number of samples:	10307
#SN	0	number of records:	22097

#callset = allel.read_vcf('test/f8_b2_4x5.vcf')


# In[26]:


sorted(callset.keys())


# In[27]:


callset['variants/ID']


# In[28]:


callset['samples'].shape


# In[29]:


callset['samples']


# In[30]:


callset['calldata/GT']


# In[31]:


haps = {}
samples = callset['samples']


# In[32]:


for position in callset['calldata/GT']:
    for person_index in range(len(position)):
        if samples[person_index]+"_A" not in haps.keys():
            haps[samples[person_index]+"_A"] = ""
        if samples[person_index]+"_B" not in haps.keys():
            haps[samples[person_index]+"_B"] = ""
        haps[samples[person_index]+"_A"]+=" "+str(position[person_index][0])
        haps[samples[person_index]+"_B"]+=" "+str(position[person_index][1])


# In[33]:


lines = []
for key,value in haps.items():
    lines.append("Real "+key+value+"\n")


# In[34]:


with open("f8_brca2_2504samples_805snps.hapt","w") as file1:
    file1.writelines(lines)
    file1.close()


# In[47]:


lines[0].replace(" ","")


# In[55]:


print(len(lines[0].replace(" ","")) - len('RealNWD100018_A'))


# In[51]:

print(len(lines))


# In[ ]:




