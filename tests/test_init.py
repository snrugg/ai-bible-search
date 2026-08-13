"""Tests for bible_study -- package entry point."""


class TestMain:
    """The main() wrapper around the Click group."""

    def test_main_is_callable(self):
        from bible_study import main
        assert callable(main)

    def test_cli_is_callable(self):
        from bible_study.cli import cli
        assert callable(cli)

    def test_main_invokes_cli_group(self, mocker):
        import bible_study
        mock_cli = mocker.patch.object(bible_study, "cli")
        bible_study.main()
        mock_cli.assert_called_once_with()

    def test_all_exports_exist(self):
        import bible_study
        assert bible_study.__all__ == ["cli", "main"]
