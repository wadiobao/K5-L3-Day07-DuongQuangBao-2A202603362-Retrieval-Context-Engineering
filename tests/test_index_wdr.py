import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
from advanced.index_wdr import build_index
from baseline import TfidfIndex


class WdrIndexTests(unittest.TestCase):
    def test_visual_notes_are_separate_retrievable_chunks(self):
        import pymupdf

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / 'sample.pdf'
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), 'Extracted paragraph.')
            document.save(pdf_path)
            document.close()
            notes_path = root / 'visual_notes.json'
            notes_path.write_text(json.dumps([{
                'pdf_page': 1,
                'printed_page': 1,
                'element': 'Figure 1',
                'kind': 'chart',
                'region': 'center',
                'description': 'The blue series rises sharply after 2020.',
            }]), encoding='utf-8')

            chunks, _ = build_index(pdf_path, notes_path)

            self.assertEqual(len(chunks), 2)
            visual_chunk = chunks[1]
            self.assertIn('The blue series rises sharply after 2020.', visual_chunk['text'])
            self.assertEqual(visual_chunk['metadata']['visual_elements'][0]['element'], 'Figure 1')
            self.assertFalse(visual_chunk['metadata']['contains_extracted_text'])
            result = TfidfIndex(chunks).search('When does the blue series rise?', top_k=1)
            self.assertEqual(result[0]['id'], visual_chunk['id'])


if __name__ == '__main__':
    unittest.main()
