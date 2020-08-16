import numpy as np
import torch 

def confusion_matrix(top_class,labels,equals,n_cat,device):
    n=len(top_class)
    count_predicted=torch.zeros(n_cat).to(device)
    count_equals=torch.zeros(n_cat).to(device)
    total_categories=torch.zeros(n_cat).to(device)
    for index in range(n):
#        print("{} {} {}".format(top_class[index],labels[index],equals[index]))
        if equals[index]==True:
            count_equals[int(top_class[index])]+=1
        count_predicted[int(top_class[index])]+=1
        total_categories[int(labels[index])]+=1
    print("Cat\tEquals\tPredicted\tTotal\tRecall\tPrecision")
    for category in range(n_cat):
        print("{}\t{}\t{}\t{}\t{}\t{}".format(category,count_equals[category],count_predicted[category],
                     total_categories[category],count_equals[category]/count_predicted[category],
                     count_equals[category]/total_categories[category]))
#    return count_equals      
def generating_indices(file_list,ini,max_files):
   n_cat=2
   a=range(n_cat)
   min_len=int(max_files/n_cat)
   indices=np.tile(a,min_len)
   reverse_dict_categories={0:'benign',1:'malignant'}
   data_body=file_list.groupby("benign_malignant")
   filenames=[]
   for n in range(min_len):
       for key, value in reverse_dict_categories.items() :
                filenames.append(data_body.get_group(value).values[n][0])

   return indices,filenames 
    	  
