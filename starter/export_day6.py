"""Export an existing Day 6 pgvector strategy into the Day 7 chunk contract."""
import argparse
import json
import os
from pathlib import Path


def export_chunks(database_url: str, dimensions: int = 768, strategy: str = 'C') -> list[dict]:
    if not database_url.strip():
        raise ValueError('DATABASE_URL chưa được cấu hình')
    if type(dimensions) is not int or dimensions <= 0:
        raise ValueError('dimensions phải là số nguyên dương')
    if not strategy.strip():
        raise ValueError('strategy không được để trống')
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError('Thiếu psycopg; cài starter/requirements.txt') from exc

    table_name = f'lab_chunks_{dimensions}'
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        rows = connection.execute(
            f"""
            SELECT chunk_id, document_id, source, page, content, metadata
            FROM {table_name}
            WHERE strategy = %s
            ORDER BY document_id, page, chunk_id
            """,
            (strategy,),
        ).fetchall()
    if not rows:
        raise RuntimeError(f'Không tìm thấy chunk trong {table_name} cho strategy {strategy}')
    return [
        {
            'id': row['chunk_id'],
            'source_id': row['document_id'],
            'source': row['source'],
            'page': row['page'],
            'text': row['content'],
            'metadata': row['metadata'],
        }
        for row in rows
    ]


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description='Export chunk từ vector store Day 6')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--strategy', default=os.getenv('DAY6_STRATEGY', 'C'))
    parser.add_argument('--dimensions', type=int, default=int(os.getenv('EMBEDDING_DIMENSIONS', '768')))
    args = parser.parse_args()
    rows = export_chunks(os.getenv('DATABASE_URL', ''), args.dimensions, args.strategy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Đã export {len(rows)} chunks ({args.strategy}) tới {args.output}')


if __name__ == '__main__':
    main()
