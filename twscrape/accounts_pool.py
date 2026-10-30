import asyncio
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Literal, TypedDict

from .account import Account, has_required_cookies
from .db import execute, fetchall, fetchone
from .http import HttpStatusError
from .logger import logger
from .login import LoginConfig, login
from .utils import get_env_bool, parse_cookies, sqlite_datetime, utc

NoAccountCause = Literal[
    "no_accounts_added",
    "all_rate_limited",
    "all_in_use",
    "all_banned",
    "all_xclid_errors",
    "all_unknown_errors",
]

# lock causes that mean "the request path is broken", not "this account is tired"
BREAKAGE_CAUSES = ("xclid_account", "xclid_parse", "gql_outdated")


class NoAccountError(Exception):
    def __init__(
        self,
        queue: str,
        cause: NoAccountCause = "all_unknown_errors",
        *,
        next_unlock_at: datetime | None = None,
        message: str | None = None,
    ):
        self.queue = queue
        self.cause = cause
        self.next_unlock_at = next_unlock_at
        super().__init__(message or self._build_message())

    def _build_message(self) -> str:
        msg = f"No account available for queue {self.queue}: {self.cause}"
        if self.next_unlock_at:
            msg += f" until {self.next_unlock_at.isoformat()}"
        return msg


class AccountInfo(TypedDict):
    username: str
    login_method: str
    logged_in: bool
    active: bool
    locks: dict[str, datetime]
    lock_causes: dict[str, str]
    last_used: datetime | None
    total_req: int
    error_msg: str | None
    last_error_cause: str | None
    last_error_status: int | None
    last_error_body: str | None


def guess_delim(line: str):
    lp, rp = tuple([x.strip() for x in line.split("username")])
    return rp[0] if not lp else lp[-1]


class AccountsPool:
    # _order_by: str = "RANDOM()"
    _order_by: str = "username"

    def __init__(
        self,
        db_file="accounts.db",
        login_config: LoginConfig | None = None,
        raise_when_no_account=False,
        wait_timeout: float | None = None,
        wait_interval: float = 5.0,
    ):
        self._db_file = db_file
        self._login_config = login_config or LoginConfig()
        self._raise_when_no_account = raise_when_no_account
        # When every active account is momentarily locked (in-use or rate-limited),
        # get_for_queue_or_wait polls for up to wait_timeout seconds (every
        # wait_interval) before giving up. This lets a brief in-use lock resolve into
        # a successful acquisition instead of an instant failure, while a real
        # rate-limit is not waited out indefinitely. None keeps the legacy behaviour
        # (raise immediately if raise_when_no_account, otherwise block forever).
        self._wait_timeout = wait_timeout
        self._wait_interval = wait_interval

    @staticmethod
    def _usernames_where(usernames: list[str]) -> tuple[str, dict[str, str]]:
        params = {f"username_{i}": x for i, x in enumerate(dict.fromkeys(usernames))}
        return ",".join(f":{x}" for x in params), params

    async def load_from_file(self, filepath: str, line_format: str):
        line_delim = guess_delim(line_format)
        tokens = line_format.split(line_delim)

        required = {"username", "password", "email", "email_password"}
        if not required.issubset(tokens):
            raise ValueError(f"Invalid line format: {line_format}")

        accounts = []
        with open(filepath) as f:
            lines = f.read().split("\n")
            lines = [x.strip() for x in lines if x.strip()]

            for line in lines:
                data = [x.strip() for x in line.split(line_delim)]
                if len(data) < len(tokens):
                    raise ValueError(f"Invalid line: {line}")

                data = data[: len(tokens)]
                vals = {k: v for k, v in zip(tokens, data) if k != "_"}
                accounts.append(vals)

        for x in accounts:
            await self.add_account(**x)

    async def add_account(
        self,
        username: str,
        password: str,
        email: str,
        email_password: str,
        user_agent: str | None = None,
        proxy: str | None = None,
        cookies: str | None = None,
        mfa_code: str | None = None,
    ):
        qs = "SELECT * FROM accounts WHERE username = :username"
        rs = await fetchone(self._db_file, qs, {"username": username})
        if rs:
            logger.warning(f"Account {username} already exists")
            return

        account = Account(
            username=username,
            password=password,
            email=email,
            email_password=email_password,
            user_agent=user_agent or "@chrome",
            active=False,
            locks={},
            stats={},
            headers={},
            cookies=parse_cookies(cookies) if cookies else {},
            proxy=proxy,
            mfa_code=mfa_code,
        )

        if has_required_cookies(account.cookies):
            account.active = True

        await self.save(account)
        logger.info(f"Account {username} added successfully (active={account.active})")

    async def add_account_cookies(self, username: str, cookies: str):
        parsed = parse_cookies(cookies)
        if not has_required_cookies(parsed):
            raise ValueError("Cookies must include auth_token and ct0")

        qs = """
        INSERT INTO accounts (username, password, email, email_password, user_agent, active, cookies)
        VALUES (:username, '_', '_', '_', '@chrome', true, :cookies)
        ON CONFLICT(username) DO UPDATE SET
            cookies = excluded.cookies,
            headers = '{}',
            active = true,
            error_msg = NULL
        """
        await execute(self._db_file, qs, {"username": username, "cookies": json.dumps(parsed)})
        logger.info(f"Cookies for account {username} updated successfully")

    async def delete_accounts(self, usernames: str | list[str]):
        usernames = usernames if isinstance(usernames, list) else [usernames]
        if not usernames:
            logger.warning("No usernames provided")
            return

        placeholders, params = self._usernames_where(usernames)
        qs = f"DELETE FROM accounts WHERE username IN ({placeholders})"
        await execute(self._db_file, qs, params)

    async def delete_inactive(self):
        qs = "DELETE FROM accounts WHERE active = false"
        await execute(self._db_file, qs)

    async def get(self, username: str):
        qs = "SELECT * FROM accounts WHERE username = :username"
        rs = await fetchone(self._db_file, qs, {"username": username})
        if not rs:
            raise ValueError(f"Account {username} not found")
        return Account.from_rs(rs)

    async def get_all(self):
        qs = "SELECT * FROM accounts"
        rs = await fetchall(self._db_file, qs)
        return [Account.from_rs(x) for x in rs]

    async def get_account(self, username: str):
        qs = "SELECT * FROM accounts WHERE username = :username"
        rs = await fetchone(self._db_file, qs, {"username": username})
        if not rs:
            return None
        return Account.from_rs(rs)

    async def save(self, account: Account):
        data = account.to_rs()
        cols = list(data.keys())

        qs = f"""
        INSERT INTO accounts ({",".join(cols)}) VALUES ({",".join([f":{x}" for x in cols])})
        ON CONFLICT(username) DO UPDATE SET {",".join([f"{x}=excluded.{x}" for x in cols])}
        """
        await execute(self._db_file, qs, data)

    async def login(self, account: Account):
        try:
            await login(account, cfg=self._login_config)
            logger.info(f"Logged in to {account.username} successfully")
            return True
        except HttpStatusError as e:
            rep = e.response
            logger.error(f"Failed to login '{account.username}': {rep.status_code} - {rep.text}")
            return False
        except Exception as e:
            logger.error(f"Failed to login '{account.username}': {e}")
            return False
        finally:
            await self.save(account)

    # Cookie accounts use password="_" and get their session through add_cookie.
    # Password accounts get their session through login/login_all.
    # Relogin resets only password accounts; cookie accounts wait for another add_cookie.
    async def login_all(self, usernames: list[str] | None = None):
        if usernames is not None and not usernames:
            return {"total": 0, "success": 0, "failed": 0}

        params = None
        if usernames is None:
            qs = "SELECT * FROM accounts WHERE active = false AND error_msg IS NULL"
        else:
            placeholders, params = self._usernames_where(usernames)
            qs = f"SELECT * FROM accounts WHERE username IN ({placeholders})"

        rs = await fetchall(self._db_file, qs, params)
        accounts = [Account.from_rs(rs) for rs in rs]
        accounts = [x for x in accounts if x.login_method == "password"]
        # await asyncio.gather(*[login(x) for x in self.accounts])

        counter = {"total": len(accounts), "success": 0, "failed": 0}
        for i, x in enumerate(accounts, start=1):
            logger.info(f"[{i}/{len(accounts)}] Logging in {x.username} - {x.email}")
            status = await self.login(x)
            counter["success" if status else "failed"] += 1
        return counter

    async def relogin(self, usernames: str | list[str]):
        usernames = usernames if isinstance(usernames, list) else [usernames]
        if not usernames:
            logger.warning("No usernames provided")
            return

        placeholders, params = self._usernames_where(usernames)
        qs = f"""
        UPDATE accounts SET
            active = false,
            locks = json_object(),
            last_used = NULL,
            error_msg = NULL,
            headers = json_object(),
            cookies = json_object(),
            user_agent = "@chrome"
        WHERE username IN ({placeholders}) AND password != '_'
        """

        await execute(self._db_file, qs, params)
        await self.login_all(usernames)

    async def relogin_failed(self):
        qs = "SELECT username FROM accounts WHERE active = false AND error_msg IS NOT NULL"
        rs = await fetchall(self._db_file, qs)
        await self.relogin([x["username"] for x in rs])

    async def reset_locks(self, keep_rate_limited: bool = False):
        """Clear queue locks.

        A crashed process leaves its 15-minute in-use leases behind; call this at
        startup. keep_rate_limited leaves, per queue, the locks X itself set with
        a rate limit (they expire on X's schedule, clearing them wastes a request)
        and drops everything else.
        """
        if not keep_rate_limited:
            await execute(
                self._db_file,
                "UPDATE accounts SET locks = json_object(), lock_causes = json_object()",
            )
            return
        for acc in await self.get_all():
            keep = {q: t for q, t in acc.locks.items() if acc.lock_causes.get(q) == "rate_limit"}
            if keep == acc.locks:
                continue
            await execute(
                self._db_file,
                "UPDATE accounts SET locks = :locks, lock_causes = :causes WHERE username = :username",
                {
                    "username": acc.username,
                    # SQLite's datetime() text form ('YYYY-MM-DD HH:MM:SS'): get_for_queue
                    # compares lock strings against datetime('now') lexically, so an ISO
                    # 'T' separator would sort after every SQLite timestamp of the same day
                    "locks": json.dumps({q: sqlite_datetime(t) for q, t in keep.items()}),
                    "causes": json.dumps(dict.fromkeys(keep, "rate_limit")),
                },
            )

    async def set_active(self, username: str, active: bool):
        qs = "UPDATE accounts SET active = :active WHERE username = :username"
        await execute(self._db_file, qs, {"username": username, "active": active})

    async def lock_until(
        self,
        username: str,
        queue: str,
        unlock_at: int,
        req_count=0,
        *,
        cause: str | None = None,
        error_status: int | None = None,
        error_body: str | None = None,
        error_msg: str | None = None,
    ):
        qs = f"""
        UPDATE accounts SET
            locks = json_set(locks, '$.{queue}', datetime({unlock_at}, 'unixepoch')),
            lock_causes = json_set(lock_causes, '$.{queue}', :lock_cause),
            stats = json_set(stats, '$.{queue}', COALESCE(json_extract(stats, '$.{queue}'), 0) + {req_count}),
            last_used = datetime({utc.ts()}, 'unixepoch'),
            error_msg = COALESCE(:error_msg, error_msg),
            last_error_cause = :cause,
            last_error_status = :error_status,
            last_error_body = :error_body
        WHERE username = :username
        """
        await execute(
            self._db_file,
            qs,
            {
                "username": username,
                "cause": cause,
                "lock_cause": cause or "unknown",
                "error_status": error_status,
                "error_body": error_body,
                "error_msg": error_msg,
            },
        )

    async def unlock(self, username: str, queue: str, req_count=0):
        qs = f"""
        UPDATE accounts SET
            locks = json_remove(locks, '$.{queue}'),
            lock_causes = json_remove(lock_causes, '$.{queue}'),
            stats = json_set(stats, '$.{queue}', COALESCE(json_extract(stats, '$.{queue}'), 0) + {req_count}),
            last_used = datetime({utc.ts()}, 'unixepoch')
        WHERE username = :username
        """
        await execute(self._db_file, qs, {"username": username})

    async def _get_and_lock(self, queue: str, condition: str) -> Account | None:
        # if space in condition, it's a subquery, otherwise it's username
        condition = f"({condition})" if " " in condition else f"'{condition}'"

        # Taking the lease drops this queue's cause: the lock now means "in
        # use", not "failed". Other queues' causes (a real rate limit on
        # SearchTimeline while we lease TweetDetail) are untouched.
        if int(sqlite3.sqlite_version_info[1]) >= 35:
            qs = f"""
            UPDATE accounts SET
                locks = json_set(locks, '$.{queue}', datetime('now', '+15 minutes')),
                lock_causes = json_remove(lock_causes, '$.{queue}'),
                last_used = datetime({utc.ts()}, 'unixepoch')
            WHERE username = {condition}
            RETURNING *
            """
            rs = await fetchone(self._db_file, qs)
        else:
            tx = uuid.uuid4().hex
            qs = f"""
            UPDATE accounts SET
                locks = json_set(locks, '$.{queue}', datetime('now', '+15 minutes')),
                lock_causes = json_remove(lock_causes, '$.{queue}'),
                last_used = datetime({utc.ts()}, 'unixepoch'),
                _tx = '{tx}'
            WHERE username = {condition}
            """
            await execute(self._db_file, qs)

            qs = f"SELECT * FROM accounts WHERE _tx = '{tx}'"
            rs = await fetchone(self._db_file, qs)

        return Account.from_rs(rs) if rs else None

    async def get_for_queue(self, queue: str) -> Account | None:
        q = f"""
        SELECT username FROM accounts
        WHERE active = true AND (
            locks IS NULL
            OR json_extract(locks, '$.{queue}') IS NULL
            OR json_extract(locks, '$.{queue}') < datetime('now')
        )
        ORDER BY {self._order_by}
        LIMIT 1
        """

        return await self._get_and_lock(queue, q)

    async def get_for_queue_or_wait(self, queue: str) -> Account | None:
        start = utc.now()
        msg_shown = False
        while True:
            account = await self.get_for_queue(queue)
            if account is not None:
                if msg_shown:
                    logger.info(f"Continuing with account {account.username} on queue {queue}")
                return account

            raise_no_account = self._raise_when_no_account or get_env_bool(
                "TWS_RAISE_WHEN_NO_ACCOUNT"
            )

            # next_available_at returns None only when no active account exists at all,
            # in which case waiting is futile. A timestamp means active accounts exist
            # but are all locked right now (in-use or rate-limited).
            nat = await self.next_available_at(queue)
            no_active = not nat

            # Give up (raise / stop) when: no active account exists, the caller opted
            # out of waiting (wait_timeout is None), or the wait budget is exhausted.
            elapsed = (utc.now() - start).total_seconds()
            give_up = no_active or self._wait_timeout is None or elapsed >= self._wait_timeout
            if give_up:
                if raise_no_account:
                    availability = await self.account_availability(queue)
                    raise NoAccountError(
                        queue,
                        availability["cause"],
                        next_unlock_at=availability["next_unlock_at"],
                    )
                if no_active:
                    logger.warning("No active accounts. Stopping...")
                    return None
                if self._wait_timeout is not None:
                    return None
                # wait_timeout is None and not raising: fall through to the legacy
                # unbounded wait below.

            if not msg_shown:
                logger.info(f'No account available for queue "{queue}". Next available at {nat}')
                msg_shown = True

            await asyncio.sleep(self._wait_interval)

    async def account_availability(self, queue: str) -> dict:
        """Why get_for_queue returned nothing: {"cause", "next_unlock_at", "accounts"}.

        A lock with no entry in lock_causes is an in-use lease (see
        _get_and_lock), so two busy accounts report all_in_use, not an error.
        Locks from XClId / GQL breakage report all_xclid_errors so a caller can
        tell "X changed something" from "wait a minute".
        """
        accounts = await self.get_all()
        now = utc.now()

        if not accounts:
            return {"cause": "no_accounts_added", "next_unlock_at": None, "accounts": accounts}

        active = [x for x in accounts if x.active]
        if not active:
            return {"cause": "all_banned", "next_unlock_at": None, "accounts": accounts}

        locked = []
        for account in active:
            locked_until = account.locks.get(queue)
            if locked_until and locked_until > now:
                locked.append((account, locked_until))

        if len(locked) < len(active):
            # free accounts exist, yet get_for_queue found none: unexpected
            return {"cause": "all_unknown_errors", "next_unlock_at": None, "accounts": accounts}

        next_unlock_at = min(x[1] for x in locked)
        causes = {x[0].lock_causes.get(queue) for x in locked}
        if "unknown" in causes:
            cause: NoAccountCause = "all_unknown_errors"
        elif None in causes:
            # at least one account is merely leased: wait, do not diagnose yet
            cause = "all_in_use"
        elif causes <= set(BREAKAGE_CAUSES):
            # every locked account hit XClId/GQL breakage: X changed something
            cause = "all_xclid_errors"
        else:
            cause = "all_rate_limited"
        return {"cause": cause, "next_unlock_at": next_unlock_at, "accounts": accounts}

    async def next_available_at(self, queue: str):
        qs = f"""
        SELECT json_extract(locks, '$."{queue}"') as lock_until
        FROM accounts
        WHERE active = true AND json_extract(locks, '$."{queue}"') IS NOT NULL
        ORDER BY lock_until ASC
        LIMIT 1
        """
        rs = await fetchone(self._db_file, qs)
        if rs:
            now, trg = utc.now(), utc.from_iso(rs[0])
            if trg < now:
                return "now"

            at_local = datetime.now() + (trg - now)
            return at_local.strftime("%H:%M:%S")

        return None

    async def mark_inactive(
        self,
        username: str,
        error_msg: str | None,
        *,
        cause: str | None = None,
        error_status: int | None = None,
        error_body: str | None = None,
    ):
        qs = """
        UPDATE accounts SET
            active = false,
            error_msg = :error_msg,
            last_error_cause = :cause,
            last_error_status = :error_status,
            last_error_body = :error_body,
            last_used = datetime(:last_used, 'unixepoch')
        WHERE username = :username
        """
        await execute(
            self._db_file,
            qs,
            {
                "username": username,
                "error_msg": error_msg,
                "cause": cause,
                "error_status": error_status,
                "error_body": error_body,
                "last_used": utc.ts(),
            },
        )

    async def reactivate(self, usernames: str | list[str]):
        usernames = usernames if isinstance(usernames, list) else [usernames]
        if not usernames:
            logger.warning("No usernames provided")
            return

        placeholders, params = self._usernames_where(usernames)
        qs = f"SELECT * FROM accounts WHERE username IN ({placeholders})"
        accounts = [Account.from_rs(x) for x in await fetchall(self._db_file, qs, params)]
        for x in sorted(set(usernames) - {a.username for a in accounts}):
            logger.warning(f"Account {x} not found")

        # an account without session cookies can't make requests until relogin or add_cookie
        usernames = []
        for x in accounts:
            if has_required_cookies(x.cookies):
                usernames.append(x.username)
            else:
                logger.warning(f"Account {x.username} has no session, use relogin or add_cookie")

        if not usernames:
            return

        placeholders, params = self._usernames_where(usernames)
        qs = f"UPDATE accounts SET active = true, error_msg = NULL WHERE username IN ({placeholders})"
        await execute(self._db_file, qs, params)
        logger.info(f"Reactivated accounts: {', '.join(usernames)}")

    async def stats(self):
        def locks_count(queue: str):
            return f"""
            SELECT COUNT(*) FROM accounts
            WHERE json_extract(locks, '$.{queue}') IS NOT NULL
                AND json_extract(locks, '$.{queue}') > datetime('now')
            """

        qs = "SELECT DISTINCT(f.key) as k from accounts, json_each(locks) f"
        rs = await fetchall(self._db_file, qs)
        gql_ops = [x["k"] for x in rs]

        config = [
            ("total", "SELECT COUNT(*) FROM accounts"),
            ("active", "SELECT COUNT(*) FROM accounts WHERE active = true"),
            ("inactive", "SELECT COUNT(*) FROM accounts WHERE active = false"),
            *[(f"locked_{x}", locks_count(x)) for x in gql_ops],
        ]

        qs = f"SELECT {','.join([f'({q}) as {k}' for k, q in config])}"
        rs = await fetchone(self._db_file, qs)
        return dict(rs) if rs else {}

    async def accounts_info(self):
        accounts = await self.get_all()

        items: list[AccountInfo] = []
        for x in accounts:
            item: AccountInfo = {
                "username": x.username,
                "logged_in": x.has_session,
                "login_method": x.login_method,
                "active": x.active,
                "locks": x.locks,
                "lock_causes": x.lock_causes,
                "last_used": x.last_used,
                "total_req": sum(x.stats.values()),
                "error_msg": str(x.error_msg)[0:60] if x.error_msg else None,
                "last_error_cause": x.last_error_cause,
                "last_error_status": x.last_error_status,
                "last_error_body": x.last_error_body,
            }
            items.append(item)

        old_time = datetime(1970, 1, 1).replace(tzinfo=timezone.utc)
        items = sorted(items, key=lambda x: x["username"].lower())
        items = sorted(
            items,
            key=lambda x: x["last_used"] or old_time if x["total_req"] > 0 else old_time,
            reverse=True,
        )
        items = sorted(items, key=lambda x: x["active"], reverse=True)
        # items = sorted(items, key=lambda x: x["total_req"], reverse=True)
        return items
