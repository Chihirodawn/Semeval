"""Check successful-only masks and all-fallback reporting without torch."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_native_inference_diagnostic import comparison


class Contract(unittest.TestCase):
    def setUp(self):
        self.rows=[{"ContextID":str(i),"Compound":"x y","ModAvg":v,"HeadAvg":v}
                   for i,v in enumerate((1.,3.,5.))]
        self.direct=[[1.,1.],[3.,3.],[5.,5.]]
        self.traces=[{"fallback":True,"attempts":[{"error":"incomplete_thinking"}]} for _ in range(6)]

    def test_all_fallback_is_explicit_not_a_thinking_gain(self):
        result=comparison(self.rows,self.direct,self.direct,self.traces)
        self.assertEqual(result["fallback_queries"],6)
        self.assertEqual(result["direct"],result["thinking_with_explicit_direct_fallback"])
        self.assertIsNone(result["valid_thinking_only_matched_components"]["Modifier"]["thinking"])

    def test_success_mask_is_component_specific(self):
        for i in (0,2):self.traces[i]={"fallback":False,"attempts":[{"error":None}]}
        result=comparison(self.rows,self.direct,self.direct,self.traces)
        self.assertEqual(result["valid_thinking_only_matched_components"]["Modifier"]["valid_queries"],2)
        self.assertEqual(result["valid_thinking_only_matched_components"]["Head"]["valid_queries"],0)
        self.assertEqual(result["first_attempt_valid_queries"],2)


if __name__=="__main__":unittest.main()
