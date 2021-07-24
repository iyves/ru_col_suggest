import configparser
import os
import pytest

from src.get_collocation_replacements import getPosMask, getMasked, getCollocationComponents, get_collocation_replacements

# Set up the configuration
path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, '../',
                                'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)


class TestGetCollocationReplacemens:
    @pytest.mark.parametrize('n,replaceAmt,expected', [
        (2, 1, [[0,1], [1,0]]),
        (2, 2, [[0,0]]),
        (2, 3, [[0,0]]),
        (3, 1, [[0,1,1], [1,0,1], [1,1,0]]),
        (3, 2, [[0,0,1], [0,1,0], [1,0,0]]),
        (3, 3, [[0,0,0]])
    ])
    def test_GetPosMask_ShouldReturnCorrectMask(self, n, replaceAmt, expected):
        masks = getPosMask(n, replaceAmt)
        for expect in expected:
            assert expect in masks

    @pytest.mark.parametrize('tokens,n,replaceAmt,pos,mask,expected', [
        (["рассматривать_V","школа_N","экономика_N"], 3, 1, True, "<mask>", [
            "<mask>_V школа_N экономика_N",
            "рассматривать_V <mask>_N экономика_N",
            "рассматривать_V школа_N <mask>_N",
        ]),
        (["рассматривать_V","школа_N","экономика_N"], 3, 2, True, "<mask>", [
            "рассматривать_V <mask>_N <mask>_N",
            "<mask>_V школа_N <mask>_N",
            "<mask>_V <mask>_N экономика_N",
        ]),
        (["рассматривать_V","школа_N","экономика_N"], 3, 3, True, "<mask>", [
            "<mask>_V <mask>_N <mask>_N",
        ]),
    ])
    def test_GetMaskedTrigramWithPos_ShouldReturnCorrectMask(self, tokens, n, replaceAmt, pos, mask, expected):
        masks = getMasked(tokens, n, replaceAmt, pos, mask)
        for expect in expected:
            assert expect in masks

    @pytest.mark.parametrize('tokens,n,replaceAmt,pos,mask,expected', [
        (["исследовать_V","вопрос_N"], 2, 1, True, "<mask>", [
            "<mask>_V вопрос_N",
            "исследовать_V <mask>_N",
        ]),
        (["исследовать_V","вопрос_N"], 2, 2, True, "<mask>", [
            "<mask>_V <mask>_N",
        ]),
    ])
    def test_GetMaskedBigramWithPos_ShouldReturnCorrectMask(self, tokens, n, replaceAmt, pos, mask, expected):
        masks = getMasked(tokens, n, replaceAmt, pos, mask)
        for expect in expected:
            assert expect in masks

    @pytest.mark.parametrize('tokens,n,replaceAmt,pos,mask,expected', [
        (["рассматривать","школа","экономика"], 3, 1, False, "<mask>", [
            "<mask> школа экономика",
            "рассматривать <mask> экономика",
            "рассматривать школа <mask>",
        ]),
        (["рассматривать","школа","экономика"], 3, 2, False, "<mask>", [
            "рассматривать <mask> <mask>",
            "<mask> школа <mask>",
            "<mask> <mask> экономика",
        ]),
        (["рассматривать","школа","экономика"], 3, 3, False, "<mask>", [
            "<mask> <mask> <mask>",
        ]),
    ])
    def test_GetMaskedTrigramWithoutPos_ShouldReturnCorrectMask(self, tokens, n, replaceAmt, pos, mask, expected):
        masks = getMasked(tokens, n, replaceAmt, pos, mask)
        for expect in expected:
            assert expect in masks

    @pytest.mark.parametrize('tokens,n,replaceAmt,pos,mask,expected', [
        (["исследовать","вопрос"], 2, 1, False, "<mask>", [
            "исследовать <mask>",
            "<mask> вопрос",
        ]),
        (["исследовать","вопрос"], 2, 2, False, "<mask>", [
            "<mask> <mask>",
        ]),
    ])
    def test_GetMaskedBigramWithoutPos_ShouldReturnCorrectMask(self, tokens, n, replaceAmt, pos, mask, expected):
        masks = getMasked(tokens, n, replaceAmt, pos, mask)
        for expect in expected:
            assert expect in masks

    @pytest.mark.parametrize('collocations,n,pos,expected', [
        (["рассматривать вопрос","исследовать вопрос",
          "рассматривать тема", "исследовать тема"], 2, False,
         [["рассматривать", "исследовать"], ["вопрос", "тема"]]
         ),
        (["рассматривать школа экономика",
          "рассматривать потребность экономика",
          "рассматривать университет экономика"], 3, False,
         [["рассматривать"], ["школа", "потребность", "университет"], ["экономика"]]
         ),
    ])
    def test_GetCollocationComponents_ShouldReturnComponents(self, collocations, n, pos, expected):
        res = getCollocationComponents(collocations, n, pos)
        assert len(res) == len(expected)
        for components, actual in zip(expected, res):
            assert len(components) == len(actual)
            assert all([c in actual for c in components])
