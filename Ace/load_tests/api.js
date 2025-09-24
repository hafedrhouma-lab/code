// Note that require() is a custom k6 implementation of module loading, which doesn't behave in the same way as the
// require() call in Node.js. Specifically, it only handles loading of built-in k6 modules, scripts on the local
// filesystem, and remote scripts over HTTP(S), but it does not support the Node.js module resolution algorithm.
//
// See https://k6.io/docs/using-k6/javascript-compatibility-mode/#about-require
import {SharedArray} from "k6/data";
import {randomItem} from "https://jslib.k6.io/k6-utils/1.1.0/index.js";
import {ACE_URL, UltronApiTester} from "./helpers/ace.js";
import {ID_URL, requestAccessToken} from "./helpers/sts.js";

const STAGE = __ENV.STAGE ? __ENV.STAGE : "qa";

export const options = {
    tags: {
        dh_app: "ace-load-test",
        dh_platform: "talabat",
        dh_tribe: "datascience",
        dh_squad: "fraud",
        dh_env: STAGE,
    },
    ext: {
        loadimpact: {
            name: "Ace API (" + STAGE + ")",
            projectID: 3623940
        },
    },

    // See https://k6.io/docs/testing-guides/api-load-testing/
    // See also https://k6.io/docs/testing-guides/automated-performance-testing/#guidance
    thresholds: {
        http_req_failed: [
            {threshold: 'rate < 0.01', abortOnFail: false} // HTTP errors should be less than 1%
        ],
        http_req_duration: ['p(95) < 5000'], // 95% of requests should be below 5s
    },
    stages: [
        {duration: `${__ENV.DURATION}s`, target: 50},
        // {duration: "2m", target: 50},
        // {duration: "5m", target: 250},
        // {duration: "2m", target: 50}
    ]
};

// const rankRequests = new SharedArray("rank", function () {
//     return [
//         // FIXME Include in Docker image...
//         open('../fixtures/api_requests/vendor_ranking/valid_customer_id.json'),
//         open('../fixtures/api_requests/vendor_ranking/req3.json'),
//         open('../fixtures/api_requests/vendor_ranking/req4.json'),
//         open('../fixtures/api_requests/vendor_ranking/req5.json'),
//         open('../fixtures/api_requests/vendor_ranking/req6.json'),
//     ];
// });

const ultronRankRequests = new SharedArray("rank", function () {
    return [
        open('../fixtures/api_requests/ultron/items_to_purchase.json'),
    ];
});


// More at https://k6.io/docs/using-k6/test-life-cycle/
export function setup() {
    console.log("Target environment for testing: " + STAGE);
    console.log("Identity service: " + ID_URL);
    console.log("API: " + ACE_URL);
    return requestAccessToken();
}

const aceApiTester = new UltronApiTester();

export default function (data) {
    aceApiTester.rankEndpoint(data.access_token, randomItem(ultronRankRequests));
}
