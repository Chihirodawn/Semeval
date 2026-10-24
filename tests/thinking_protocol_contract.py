"""Stdlib regression tests for leakage, admission, schedules and gates."""
import copy
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from thinking_protocol import (attach_gold, fallback_rate, identity, inputs, pairing,
                               promotion, sample_seed, validate_reasoning)


class Tokenizer:
    def encode(self,text,add_special_tokens=False):
        return list(text.encode())

    def apply_chat_template(self,messages,**kwargs):
        text=str(messages)+str(kwargs["enable_thinking"])
        return list(text.encode())


def fixture():
    return [{"ContextID":str(i),"Context":"The account book was open.","Compound":f"compound {i//4}",
             "Mod":"account","Head":"book","ModAvg":i/4,"HeadAvg":i/5} for i in range(12)]


class Contract(unittest.TestCase):
    def test_gold_is_not_an_input(self):
        rows=fixture();changed=copy.deepcopy(rows)
        for r in changed:r.update(ModAvg=987,HeadAvg=-123)
        for style in ("direct","free","evidence"):
            a=inputs(rows,Tokenizer(),style)
            self.assertEqual(a,inputs(changed,Tokenizer(),style))
            self.assertTrue(all("gold" not in r and "ModAvg" not in r for r in a))
            self.assertNotEqual(attach_gold(a,rows),attach_gold(a,changed))

    def test_admission(self):
        context="The account book was open."
        good='Meaning: a financial record\nLiteral: book retains its sense\nEvidence: "account book"\n</think>'
        self.assertIsNone(validate_reasoning(good,"evidence",context,"think_end"))
        self.assertIsNone(validate_reasoning(good[len('Meaning: '):],"evidence",context,"think_end",'Meaning: '))
        self.assertIsNotNone(validate_reasoning('</think>',"free",context,"think_end",'The focus word '))
        for text,style,stop in [(good,"evidence","token_limit"),
                                (good.replace("account book","not in sentence"),"evidence","think_end"),
                                ("The rating is five.</think>","free","think_end"),
                                ("Score: 4.2</think>","free","think_end"),
                                ("I give it 4 out of 5.</think>","free","think_end"),
                                ("</think>","free","think_end"),
                                ("Meaning: fine</think>","evidence","think_end")]:
            self.assertIsNotNone(validate_reasoning(text,style,context,stop))

    def test_native_inputs_name_words_without_role_metadata(self):
        for style in ("free","evidence"):
            records=inputs(fixture()[:1],Tokenizer(),style)
            for component,row in enumerate(records):
                prompt=bytes(row["prompt_ids"]).decode()
                word=("account","book")[component]
                self.assertIn(f'Explain the word "{word}"',prompt)
                for label in ("Component:","Modifier:","Head:","Focus word:"):
                    self.assertNotIn(label,prompt)
                self.assertEqual(row["reasoning_prefix"],"")

    def test_pairing_and_seed(self):
        records=attach_gold(inputs(fixture(),Tokenizer(),"free"),fixture())
        a,stats=pairing(records,2027,1)
        b,other=pairing([dict(r,gold=999) for r in records],2027,1)
        self.assertEqual([[identity(r) for r in g] for g in a],[[identity(r) for r in g] for g in b])
        self.assertEqual(stats,other)
        self.assertEqual(len({identity(r) for g in a for r in g}),len(records))
        self.assertTrue(all(len({r["component"] for r in g})==1 for g in a))
        self.assertNotEqual(stats["schedule_sha256"],pairing(records,2027,2)[1]["schedule_sha256"])
        self.assertEqual(sample_seed(2027,records[0]),sample_seed(2027,dict(records[0],gold=999)))
        self.assertNotEqual(sample_seed(2027,records[0]),sample_seed(2027,records[0],1))

    def test_gates(self):
        base={"mean_spearman":.60,"mean_rmse":.60}
        self.assertTrue(promotion({"mean_spearman":.61,"mean_rmse":.605},base))
        self.assertFalse(promotion({"mean_spearman":.604,"mean_rmse":.50},base))
        self.assertFalse(promotion({"mean_spearman":.61,"mean_rmse":.62},base))
        self.assertEqual(fallback_rate([{"fallback":False},{"fallback":True}]),.5)


if __name__=="__main__":unittest.main()
