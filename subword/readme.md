## 难难难，道德玄 ，不对知音不可谈 ，对了知音谈几句 ，不对知音枉废舌尖 。

## 准备数据
1. 生成数据`./data/example.txt`, 参考`./data/dataset.py`
2. 利用subword的方式生成分词文件和词典
    * **`spm_train --input=./dataset/example.txt --model_prefix=example --vocab_size=80000 --character_coverage=0.9995 --model_type=bpe`**
    * 生成模型文件**`example.model`**和**`example.vocab`**
    * 将模型放到data目录下**`mv example.* ./`**
3. 将数据进行分词操作
    * **`spm_encode --model=./data/example.model --output_format=piece < ./data/example.txt > ./data/examples_seg_words.txt`**
    * **`spm_encode --model=./data/example.model --output_format=id < ./data/example.txt > ./data/examples_seg_words_id.txt`**
    * 这里生成**`./data/examples_seg_words.txt`**即可

## 
    