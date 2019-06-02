#!/usr/bin/env bash

cd ..


#rsync -avz --exclude=".git" ./jacala/bert-tensorflow/glue_data companyai8way@10.64.50.108:/home/companyai8way/ryan/bert

#rsync -avz --exclude=".*" --exclude=".git" ./ irteam@10.108.15.73::R/home1/irteam/ryan/dssm_tf

rsync -avzP --exclude='data_out' --exclude=".git" --exclude="bert_dssm" ./ CompanyAI@10.64.48.154:/home/CompanyAI/ryan/sent_sim