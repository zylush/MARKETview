# Inquiry Contact TDD Evidence

## Source and user journey

No plan file was supplied. The journey was derived from the request: as a visitor or signed-in
user, I can find an inquiry email link on either page so I can contact the site owner.

## RED and GREEN evidence

| Stage | Command | Result | Guarantee |
| --- | --- | --- | --- |
| RED | '.venv\Scripts\pytest.exe -m e2e --no-cov -k inquiry_contact' | Failed because the login-page email link was not visible | The test exercised the missing user-visible behavior |
| GREEN | '.venv\Scripts\pytest.exe -m e2e --no-cov -k inquiry_contact' | 1 passed | Both pages show the exact inquiry sentence and use the expected mailto URL |
| Regression | '.venv\Scripts\pytest.exe -m e2e --no-cov' | 15 passed | Existing browser journeys still work |
| Coverage | '.venv\Scripts\pytest.exe -m "not e2e"' | 294 passed; 81.71% coverage | Unit and integration behavior remains above the 80% threshold |

## Checkpoint evidence

- 19c6479 records the failing browser specification.
- 81c80ff records the minimal templates and styles that made it pass.

## Additional verification and known gaps

Ruff lint/format, strict mypy, dependency audit, secret scan, and the Vercel production-target
build passed. The browser assertion covers rendered text, visibility, and the mailto destination;
it does not launch the user's local email client.
