# data downloaded from https://labshare.cshl.edu/shares/gillislab/resource/Primate_MTG_coexp/
library(SeuratObject)
library(Matrix)
library(readr)
library(Seurat)


base_path <- '/Users/kexinwang/Documents/harvard/research/contrastive_learning/python/monkey_human_data/'
human=readRDS(paste0(base_path, 'human_SCT_UMI_expression_matrix.RDS'))
human_less=subset(x = human, downsample = 10000)
human_less_normalized=NormalizeData(object=human_less)
human_data=as.data.frame(human_less_normalized@assays$RNA@data)
write.csv(human_data, "human_data.csv")


gorilla=readRDS(paste0(base_path, 'gorilla_SCT_UMI_expression_matrix.RDS'))
gorilla_less=subset(x = gorilla, downsample = 10000)
gorilla_less_normalized=NormalizeData(object=gorilla_less)
gorilla_data=as.data.frame(gorilla_less_normalized@assays$RNA@data)
write.csv(gorilla_data, "gorilla_data.csv")

chimp=readRDS(paste0(base_path, 'chimp_SCT_UMI_expression_matrix.RDS'))
chimp_less=subset(x = chimp, downsample = 10000)
chimp_less_normalized=NormalizeData(object=chimp_less)
chimp_data=as.data.frame(chimp_less_normalized@assays$RNA@data)
write.csv(chimp_data, "chimp_data.csv")
