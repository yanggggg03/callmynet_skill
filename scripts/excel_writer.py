"""Write ordinary XLSX workbooks without extra Excel dependencies."""

import json
import math
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


def _column(index):
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _cell(address, value, header=False):
    if value is None:
        return f'<c r="{address}"/>'
    if isinstance(value, bool) and not header:
        return f'<c r="{address}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)) and not header:
        if not math.isfinite(value):
            raise ValueError("Excel results must contain finite numeric values")
        return f'<c r="{address}" s="2"><v>{value:.15g}</v></c>'
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False, allow_nan=False)
    value = str(value)
    if len(value) > 32767:
        raise ValueError("Cell exceeds Excel text limit; reduce candidates or use JSON")
    if any(ord(c) < 32 and c not in "\t\n\r" for c in value):
        raise ValueError("Cell contains unsupported XML control characters")
    style = ' s="1"' if header else ""
    return f'<c r="{address}" t="inlineStr"{style}><is><t xml:space="preserve">{escape(value)}</t></is></c>'


def write_workbook(destination, sheets):
    """Sheets contain (name, headers, iterable-of-rows); destination may be BytesIO."""
    if isinstance(destination, (str, Path)):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    relationship = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    package = "http://schemas.openxmlformats.org/package/2006/relationships"
    content = "http://schemas.openxmlformats.org/package/2006/content-types"
    types = [
        f'<Types xmlns="{content}">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    sheet_tags, relations = [], []
    styles = f"""<styleSheet xmlns="{namespace}">
<numFmts count="1"><numFmt numFmtId="164" formatCode="0.000000"/></numFmts>
<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font></fonts>
<fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF17365D"/></patternFill></fill></fills>
<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/><xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs>
<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>"""
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, (name, headers, rows) in enumerate(sheets, 1):
            if not headers or len(headers) > 16384:
                raise ValueError("Invalid Excel column count")
            if not name or len(name) > 31 or any(c in name for c in "[]:*?/\\"):
                raise ValueError("Invalid Excel sheet name")
            # Stream XML so large prediction tables do not create an XML string copy.
            with archive.open(f"xl/worksheets/sheet{index}.xml", "w") as stream:

                def emit(value):
                    stream.write(value.encode("utf-8"))

                emit(
                    f'<worksheet xmlns="{namespace}"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" state="frozen" activePane="bottomLeft"/></sheetView></sheetViews><cols><col min="1" max="{len(headers)}" width="24" customWidth="1"/></cols><sheetData>'
                )
                emit(
                    '<row r="1">'
                    + "".join(
                        _cell(f"{_column(i)}1", str(v), True)
                        for i, v in enumerate(headers, 1)
                    )
                    + "</row>"
                )
                last = 1
                for last, row in enumerate(rows, 2):
                    if last > 1048576:
                        raise ValueError("Excel row limit exceeded; split the input")
                    if len(row) != len(headers):
                        raise ValueError("Excel row has incorrect column count")
                    emit(
                        f'<row r="{last}">'
                        + "".join(
                            _cell(f"{_column(i)}{last}", v)
                            for i, v in enumerate(row, 1)
                        )
                        + "</row>"
                    )
                emit(
                    f'</sheetData><autoFilter ref="A1:{_column(len(headers))}{last}"/></worksheet>'
                )
            types.append(
                f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
            sheet_tags.append(
                f'<sheet name="{escape(name, {chr(34): "&quot;"})}" sheetId="{index}" r:id="rId{index}"/>'
            )
            relations.append(
                f'<Relationship Id="rId{index}" Type="{relationship}/worksheet" Target="worksheets/sheet{index}.xml"/>'
            )
        relations.append(
            f'<Relationship Id="styles" Type="{relationship}/styles" Target="styles.xml"/>'
        )
        archive.writestr("xl/styles.xml", styles)
        archive.writestr("[Content_Types].xml", "".join(types) + "</Types>")
        archive.writestr(
            "_rels/.rels",
            f'<Relationships xmlns="{package}"><Relationship Id="rId1" Type="{relationship}/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        )
        archive.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{namespace}" xmlns:r="{relationship}"><sheets>{"".join(sheet_tags)}</sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<Relationships xmlns="{package}">{"".join(relations)}</Relationships>',
        )


def export_predictions(path, results, metadata):
    headers = (
        "source_row",
        "mag",
        "deg",
        "magnitude_unit",
        "L1",
        "L2",
        "length_unit",
        "estimated_matched_mag",
        "estimated_matched_deg",
        "estimated_return_loss_db",
        "candidate_count",
        "selected_candidate",
        "training_support_distance",
        "outside_training_input_box",
        "bounds",
        "bounds_source",
        "performance_source",
    )
    rows = ([item.get(key) for key in headers] for item in results)
    sheets = [
        ("predictions", headers, rows),
        ("说明", ("项目", "说明"), list(metadata.items())),
    ]
    if any("candidates" in item for item in results):
        candidates = []
        for row, item in enumerate(results, 1):
            for index, pair in enumerate(item.get("candidates", [])):
                candidates.append(
                    [
                        row,
                        item["source_row"],
                        index,
                        *pair,
                        item["length_unit"],
                        index == item["selected_candidate"],
                    ]
                )
        sheets.append(
            (
                "candidates",
                (
                    "result_row",
                    "source_row",
                    "candidate_index",
                    "L1",
                    "L2",
                    "length_unit",
                    "selected",
                ),
                candidates,
            )
        )
    write_workbook(path, sheets)
