from app.modules.ocr_parser.lab_ocr import LabOCRParser


def test_ocr_parser_structure():
    parser = LabOCRParser()
    out = parser.parse_bytes(b"dummy content")
    assert "Hemoglobin" in out.values
    assert out.values["WBC"].flag in {"HIGH", "LOW", "NORMAL"}
