import json
import ollama
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
language = 'English'
choices = ['Very Inaccurate', 'Moderately Inaccurate','Neither Accurate Nor Inaccurate', 'Moderately Accurate', 'Very Accurate']
TEMPERS = [
    "You are a person who values order, dutifulness, achievement striving, self-discipline, and deliberation.",
    "You are a person who values trust, straightforwardness, altruism, compliance, mosdesty and tendermindedness.",
    "You are a person who values anxiety, anger hostility, depression, self-consciousness, impulsiveness, and vulnerability.",
    "You are a person who values fantasy, aesthetics and values.", 
    "You are a person who values warmth, gregariousness, assertiveness and excitement seeking.",#red openness
    "You are a person."
]
LABELS = ['Conscientiousness', 'Agreeableness', 'Neuroticism', 'Openness', 'Extraversion','Empty']
model_name = 'llama3.2-vision:90b'  

system_prompt = """The following statement describe people's behaviours. Please select how accurately the statement describes you.  
'Very Accurate', 'Moderately Accurate', 'Neither Accurate Nor Inaccurate', 'Moderately Inaccurate', 'Very inaccurate'.
Please provide your rating, don't give any explanation for your choice."""

Allres = {}
graphdata = []  
ALLcount = []

with open('/mnt/data/guoxin/agentscope/b5-johnson-120-ipip-neo-pi-r/data/en/questions.json', 'r') as question_file:
    questions = json.load(question_file)
    for n in range(0,len(TEMPERS)):
        res = {'O':0, 'A':0, 'C':0, 'N':0, 'E':0}
        count = {'O':0, 'A':0, 'C':0, 'N':0, 'E':0}
        for question in questions:
            answer = ollama.chat(
                model = model_name,
                messages=[{
                    'role':'system',
                    'content':TEMPERS[n],
                },{
                    'role':'user',
                    'content':system_prompt+"Here is the statement:"+ question['text'],
                }]
            )
            print(question['text'],answer.message['content'])
            for choice in choices:
                if choice.lower() in answer.message['content'].lower():
                    count[question['domain']] += 1
                    if question['keyed'] == 'plus':
                        res[question['domain']] += 1 + choices.index(choice)
                    elif question['keyed'] == 'minus':
                        res[question['domain']] += 5 - choices.index(choice)
                    break
        print(res)
        print(count)
        for key in res:
            res[key] = round(res[key]/count[key],3)
        Allres.update({f'Agent {n+1}:{LABELS[n]}':res})
        graphdata.append(res)
        ALLcount.append(count)
print(Allres)
print(ALLcount)

# 数据
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
with open('Big5LLMTestFigs/'+model_name+' '+language+' '+current_time+'.json', 'w') as f:
    json.dump(Allres, f, indent=4)
data = graphdata
labels = ['O', 'A', 'C', 'N', 'E']
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合雷达图

# 创建图形
fig, ax = plt.subplots(figsize=(12,9), subplot_kw=dict(polar=True))

# 绘制每个Agent的数据
for i, agent_data in enumerate(data):
    values = [agent_data[label] for label in labels]
    values += values[:1]  # 闭合雷达图
    ax.plot(angles, values, label=f'Agent {i+1}:{LABELS[i]}')
    ax.fill(angles, values, alpha=0.25)

# 设置标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

#
ax.legend(loc='upper right',bbox_to_anchor=(0.1, 0.1)) 
# 显示图形
plt.title(model_name+f'[{language}]'+':Big Five Traits')
plt.tight_layout() 

plt.savefig('/mnt/data/guoxin/Big5LLMTestFigs/'+model_name+'_radarchart_'+current_time+'.png',dpi= 300)