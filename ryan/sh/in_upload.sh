#!/bin/bash

cd ..

#cmog 5 server

# rsync -avzP --exclude=".git" --delete ./data_in/dt=2019-03-05T14-37-22.556Z irteam@10.116.239.211::R/user/companyai/data/db/host=10.116.81.201/schema=mogli2tool/domainId=3452/dt=2019-03-05T14-37-22.556Z

# hdfs://c3v/user/companyai/data/db/host=10.116.81.201/schema=mogli2tool/domainId=3452/dt=2019-03-05T14-37-22.556Z

# rsync -avz --exclude=".git" --delete ./ irteam@10.108.15.73::R/home1/irteam/ryan/dssm
#rsync -avz --exclude=".git" ./ irteam@10.108.15.73::R/home1/irteam/ryan/dssm

#rsync -avz --exclude=".git" ./jacala/bert-tensorflow/glue_data irteam@10.108.15.73::R/home1/irteam/ryan/bert

#rsync -avz --exclude=".git" ./jacala/bert-tensorflow/glue_data companyai8way@10.64.50.108:/home/companyai8way/ryan/bert

#rsync -avz --exclude=".*" --exclude=".git" ./ irteam@10.108.15.73::R/home1/irteam/ryan/dssm_tf

rsync -avzP --exclude='data_out' --exclude=".git" --exclude="bert_dssm" --delete ./ companyai8way@10.64.50.108:/home/companyai8way/ryan/dssm


# rsync -avzP --exclude=".git" --exclude="OLD_dssm" --exclude="data_out/" ./ companyai8way@10.64.50.108:/home/companyai8way/ryan/dssm
# rsync -avzP --exclude=".git" --exclude="OLD_dssm" --exclude="data_out/" ./ companyai8way@10.64.50.108:/home/companyai8way/ryan/dssm
#rsync -avzP --exclude=".git" --delete --exclude="OLD_dssm" --exclude="data_out/./ companyai8way@10.64.50.108:/home/companyai8way/ryan/dssm"