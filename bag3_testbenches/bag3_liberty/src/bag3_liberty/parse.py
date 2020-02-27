# SPDX-License-Identifier: Apache-2.0
# Copyright 2019 Blue Cheetah Analog Design Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import List, Tuple, Sequence, Union, Optional, TextIO

import re
from pathlib import Path
from dataclasses import dataclass

from .util import remove_comments


@dataclass
class LibAttr:
    name: str
    val: str


@dataclass
class LibFun:
    name: str
    args: List[str]


@dataclass
class LibObject:
    name: str
    dtype: str
    objects: List[Union[LibAttr, LibFun, LibObject]]


# From https://stackoverflow.com/questions/2785755/
# how-to-split-but-ignore-separators-in-quoted-strings-in-python"""
comma_pat = re.compile(r'''((?:[^,"']|"[^"]*"|'[^']*')+)''')


def read_liberty(lib_file: Union[str, Path]) -> LibObject:
    with open(lib_file, 'r') as f:
        text = f.read()

    text = remove_comments(text)
    lines = _remove_white_spaces_and_continuations(text)
    stop = len(lines)
    ans, stop_idx = _parse_lib(lines, 0, stop)
    if stop_idx != stop:
        raise ValueError(f'Lib reading stopped with {stop - stop_idx} lines left')
    return ans


def write_liberty(lib_file: Union[str, Path], data: LibObject) -> None:
    if data.dtype != 'library':
        raise ValueError('Cannot write non-library objects.')

    if isinstance(lib_file, str):
        path = Path(lib_file)
    else:
        path = lib_file

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        _write_lib_object(f, data, '')


def _remove_white_spaces_and_continuations(text: str) -> Sequence[str]:
    ans = []
    lines = text.split('\n')
    start = 0
    stop = len(lines)
    while start < stop:
        line = lines[start].strip()
        if not line:
            start += 1
            continue
        while line.endswith('\\'):
            if start == stop - 1:
                line = line[:-1]
            else:
                start += 1
                line = f'{line[:-1]}{lines[start].strip()}'
        ans.append(line)
        start += 1

    return ans


def _write_lib_object(stream: TextIO, data: LibObject, tab: str) -> None:
    stream.write(f'{tab}{data.dtype} ({data.name}) {{\n')
    tab_in = tab + (' ' * 4)
    for obj in data.objects:
        if isinstance(obj, LibAttr):
            stream.write(f'{tab_in}{obj.name} : {obj.val};\n')
        elif isinstance(obj, LibFun):
            if obj.name == 'values':
                # special hack for values table
                arg_str = f', \\\n{tab_in}    '.join(obj.args)
                arg_str = f' \\\n{tab_in}    {arg_str} \\\n{tab_in}'
            else:
                arg_str = ", ".join(obj.args)
            stream.write(f'{tab_in}{obj.name} ({arg_str});\n')
        else:
            _write_lib_object(stream, obj, tab_in)
    stream.write(f'{tab}}}\n')


def _parse_lib(lines: Sequence[str], start: int, stop: int
               ) -> Tuple[Optional[Union[LibObject, LibAttr, LibFun]], int]:
    cur_line = lines[start].strip()
    while not cur_line and start < stop:
        start += 1
        cur_line = lines[start].strip()

    if not cur_line:
        raise ValueError(f'Cannot find closing brackets between lines {start + 1} and {stop}')

    if cur_line == '}':
        return None, start + 1

    is_statement = _is_line_statement(cur_line)
    if is_statement is None:
        raise ValueError(f'Failed to parse line number {start + 1}:\n{cur_line}')

    if is_statement:
        # single statement line
        return _parse_single_statement(cur_line, start), start + 1

    arr = cur_line[:-1].split('(', maxsplit=1)
    if len(arr) == 1:
        raise ValueError(f'Failed to parse line number {start + 1}:\n{cur_line}')

    obj_type = arr[0].strip()
    obj_name = arr[1].strip()
    if obj_name[-1] != ')':
        raise ValueError(f'Failed to parse line number {start + 1}:\n{cur_line}')
    obj_name = obj_name[:-1].strip()

    return _parse_lib_object(obj_name, obj_type, lines, start + 1, stop)


def _parse_lib_object(name: str, dtype: str, lines: Sequence[str], start: int, stop: int
                      ) -> Tuple[LibObject, int]:
    objects = []
    ans = LibObject(name=name, dtype=dtype, objects=objects)
    while start < stop:
        cur_obj, start = _parse_lib(lines, start, stop)
        if cur_obj is None:
            # read the end of this object
            return ans, start
        else:
            objects.append(cur_obj)
    return ans, start


def _is_line_statement(line: str) -> Optional[bool]:
    if line[-1] == ';':
        return True
    if line[-1] != '{':
        if ':' in line:
            print(f'WARNING: adding semi-colon to the following line:\n{line}')
            return True
        return None
    return False


def _parse_single_statement(line: str, line_idx: int) -> Union[LibAttr, LibFun]:
    if line[-1] == ';':
        # remove semi-colon
        line = line[:-1].strip()
    if ':' in line:
        # simple attribute
        arr = line.split(':', maxsplit=1)
        return LibAttr(name=arr[0].strip(), val=arr[1].strip())
    elif line[-1] == ')':
        # function
        arr = line.split('(', maxsplit=1)
        name = arr[0].strip()
        values = comma_pat.split(arr[1][:-1])[1::2]
        return LibFun(name=name, args=[v.strip() for v in values])
    else:
        raise ValueError(f'Failed to parse line number {line_idx + 1}:\n{line}')
