import configparser
import os
import pytest

from src.collocation_attestor import CollocationAttestor

# Set up the configuration
path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, '../',
                                'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)


class TestCollocationAttestor:
    def test_ConnectToAttestor_ShouldSuccessfullyConnect(self):
        with CollocationAttestor() as attestor:
            assert attestor is not None
            assert attestor.connection is not None

    def test_GetDomainSize_ShouldReturnCorrectDomainSize(self):
        with CollocationAttestor() as attestor:
            assert attestor is not None
            assert attestor.connection is not None
            assert attestor._domain_size == -1
            assert attestor.get_domain_size() == 978005

    def test_GetFrequencyEmptyList_ShouldReturnEmptyList(self):
        with CollocationAttestor() as attestor:
            assert attestor is not None
            assert attestor.connection is not None
            assert len(attestor.get_frequency([])) == 0

    @pytest.mark.parametrize('input,expected_output', [
        (['что'], [914458]),
        (['что', 'делать'], [914458, 88138]),
        (['что делать'], [3694]),
        (['что делать', 'рассматривать дело'], [3694, 721]),
        (['рассматривать потребность экономика'], [31]),
        (['рассматривать потребность экономика', 'рассматривать школа экономика'], [31, 6]),
    ])
    def test_GetFrequencyExisting_ShouldReturnCorrectFrequency(self, input, expected_output):
        with CollocationAttestor() as attestor:
            assert attestor is not None
            assert attestor.connection is not None
            freqs = {}
            for row in attestor.get_frequency(input):
               freqs[row[0].strip()] = int(row[1])
            for colloc, expected_freq in zip(input, expected_output):
                assert colloc in freqs
                assert freqs[colloc] == expected_freq
                # check that the value was caches
                assert colloc in attestor._collocation_stats
                assert "freq" in attestor._collocation_stats[colloc]
                assert attestor._collocation_stats[colloc]["freq"] == expected_freq


    @pytest.mark.parametrize('input,expected_output', [
        (['вдаыова'], [0]),
        (['вдаыова', 'даовлаоадв'], [0, 0]),
        (['чвдаыовато делать'], [0]),
        (['чвдаыовато делать', 'радаовлаоадвссматривать дело'], [0, 0]),
        (['равдаыовассматривать потребность экономика'], [0]),
        (['равдаыовассматривать потребность экономика', 'радаовлаоадвссматривать школа экономика'], [0, 0]),
    ])
    def test_GetFrequencyNonexisting_ShouldReturn0Frequency(self, input, expected_output):
        with CollocationAttestor() as attestor:
            assert attestor is not None
            assert attestor.connection is not None
            freqs = {}
            for row in attestor.get_frequency(input):
                freqs[row[0].strip()] = int(row[1])
            for colloc, expected_freq in zip(input, expected_output):
                assert colloc in freqs
                assert freqs[colloc] == expected_freq

    def test_GetFrequency4gram_ShouldReturnEmptyList(self):
        with CollocationAttestor() as attestor:
            assert attestor is not None
            assert attestor.connection is not None
            assert len(attestor.get_frequency(["раз два три четыре"])) == 0

    def test_AttestEmptyList_ShouldReturnEmptyList(self):
        with CollocationAttestor() as attestor:
            assert attestor is not None
            assert attestor.connection is not None
            assert len(attestor.attest_collocations([])) == 0

    @pytest.mark.parametrize('input,expected_output', [
        ([['мочь'],['быть']], ['мочь быть']),
        ([['мочь','что'],['быть','делать']], ['мочь быть','мочь делать','что быть','что делать']),
        ([['рассматривать'],['школа'],['экономика']], ['рассматривать школа экономика']),
        ([['рассматривать','регулирование'],['школа','потребность'],['экономика','правопонимание']],
            ['рассматривать школа экономика', 'рассматривать потребность экономика',
             'рассматривать потребность правопонимание', 'регулирование школа экономика']),
    ])
    def test_AttestExisting_ShouldAttestAll(self, input, expected_output):
        with CollocationAttestor() as attestor:
            assert attestor is not None
            assert attestor.connection is not None
            attested = attestor.attest_collocations(input)
            for expected in expected_output:
                assert expected in attested

    @pytest.mark.parametrize('input', [
        ([['коллокация'],['ураураура']]),
        ([['зачем'],['мне'],['ураураура']])
    ])
    def test_AttestNonexisting_ShouldReturnEmptyList(self, input):
        with CollocationAttestor() as attestor:
            assert attestor is not None
            assert attestor.connection is not None
            assert len(attestor.attest_collocations(input)) == 0

    @pytest.mark.parametrize('input', [
        ([['что']]),
        ([['раз'],['два'],['три'], ['четыре']])
    ])
    def test_AttestNonbitrigram_ShouldReturnEmptyList(self, input):
        with CollocationAttestor() as attestor:
            assert attestor is not None
            assert attestor.connection is not None
            assert len(attestor.attest_collocations(input)) == 0

    def test_GetStatsEmptyList_ShouldReturnEmptyDictionary(self):
        with CollocationAttestor() as attestor:
            assert attestor is not None
            assert attestor.connection is not None
            stats = attestor.get_collocation_stats([])
            assert isinstance(stats, dict)
            assert len(stats) == 0

    @pytest.mark.parametrize('input', [
        (['что', 'это']),
        (['раз два три четыре'])
    ])
    def test_GetStatsNonbitrigram_ShouldReturnEmptyDictionary(self, input):
        with CollocationAttestor() as attestor:
            assert attestor is not None
            assert attestor.connection is not None
            stats = attestor.get_collocation_stats([])
            assert isinstance(stats, dict)
            assert len(stats) == 0

    @pytest.mark.parametrize('input,expected_outcome', [
        (['что делать'], [0]),
        (['рассматривать школа экономика'], [0])
    ])
    def test_GetStatsExistingCollocations_ShouldStats(self, input, expected_outcome):
        with CollocationAttestor() as attestor:
            assert attestor is not None
            assert attestor.connection is not None
            stats = attestor.get_collocation_stats(input)
            assert isinstance(stats, dict)
            for collocation in input:
                assert collocation in stats
                assert "pmi" in stats[collocation]
                assert "t_score" in stats[collocation]
                assert "ngram_freq" in stats[collocation]
                # Checking the cache
                assert collocation in attestor._collocation_stats
                assert "freq" in attestor._collocation_stats[collocation]
                assert "t_score" in attestor._collocation_stats[collocation]
                assert "pmi" in attestor._collocation_stats[collocation]
                # TODO: make sure the pmi and t-score calculations are correct
