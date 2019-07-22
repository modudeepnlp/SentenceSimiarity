#!/bin/bash

cd ..

#cai gpuls
#rsync -avzP companyai8way@10.64.50.108:/home/companyai8way/ryan/dssm/data_out/ ./data_out/
rsync -avzP CompanyAI@10.64.48.154:/home/CompanyAI/ryan/dssm/data ./data
#scp companyai8way@10.64.48.154:/home/CompanyAI/ryan/dssm/data ./data
# ssh-copy-id -i ~/.ssh/id_rsa.pub CompanyAI@10.64.48.154