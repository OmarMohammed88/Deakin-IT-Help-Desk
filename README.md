# Deakin IT Help Desk - RAG Application

This repository contains the implementation of a **Retrieval-Augmented Generation (RAG)** application designed for the **Deakin IT Help Desk**. The app is used for automating and improving the process of responding to IT-related queries.

## Table of Contents

- [Installation](#installation)
- [Download Database and Datasets](#download-database-and-datasets)
- [Running the Application](#running-the-application)

## Installation

To set up the environment for running the RAG application, follow these steps:

### 1. Load Anaconda module
First, load the Anaconda3 module:

```bash
module load Anaconda3

conda env create -f environment.yml

conda source

conda activate rag_app
```
## Download Database and Datasets
Ensure the application functions correctly, you need to download the necessary database and datasets.
- Download the required vector database files from this Google Drive link.
    - [Vector_Database](https://drive.google.com/drive/folders/1O3B4Ud1s3nJFq8LCW_4TsimMXo5xWniO?usp=drive_link)

    - [Datasets](https://drive.google.com/drive/folders/1EDnGRVnUS7HEuUPpsPwPgbmQqs20fxsq?usp=drive_link)


## Running the Application
Once the environment is set up and the necessary data is downloaded, you can run the application with the following command:

```
sbatch run_rag
```
