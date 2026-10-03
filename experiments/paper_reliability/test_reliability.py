import unittest
import numpy as np
from reliability import groups,pair_vectors,metrics

class Tests(unittest.TestCase):
 def test_transitive_grouping(self):
  raw=[{'idx':i,'func':str(i),'project':'p','commit_id':c} for i,c in enumerate(['a','b','b','c','z','z'])]
  g=groups(raw,'commit');self.assertEqual(len(set(g[:4])),1);self.assertNotEqual(g[0],g[4])
  raw[4]['idx']=0;self.assertEqual(len(set(groups(raw,'commit'))),1)
 def test_unknown_keeps_pair_out(self):
  v,mask=pair_vectors(np.array([1,0,1,0]),np.array([1,0,1,np.nan]));self.assertEqual(mask.tolist(),[True,False])
  f,pc=metrics(v[mask].sum(0));self.assertEqual(f,1);self.assertEqual(pc,1)
 def test_two_yes_not_pair_correct(self):
  v,m=pair_vectors(np.array([1,0]),np.array([1,1]));f,pc=metrics(v.sum(0));self.assertAlmostEqual(f,2/3);self.assertEqual(pc,0)
if __name__=='__main__':unittest.main()
